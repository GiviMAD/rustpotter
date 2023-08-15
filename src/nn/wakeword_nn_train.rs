use super::wakeword_nn::{flat_features, get_tensors_data, new_model_impl, Mlp, ModelImpl};
use crate::{
    constants::FEATURE_EXTRACTOR_NUM_FEATURES, wakeword_model::ModelWeights, WakewordModel,
};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, VarMap};
use std::{
    collections::HashMap,
    fs::{self},
    io::{BufReader, Error, ErrorKind},
};

pub trait TrainableWakeword {
    fn train_from_sample_buffers(
        samples: HashMap<String, Vec<u8>>,
        test_samples: HashMap<String, Vec<u8>>,
        learning_rate: f64,
        epochs: usize,
        wakeword_model: Option<WakewordModel>,
    ) -> Result<WakewordModel, Error> {
        if samples.is_empty() {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                "No training data provided",
            ));
        }
        if test_samples.is_empty() {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                "No test data provided",
            ));
        }
        let mut labels: Vec<String> = wakeword_model
            .as_ref()
            .map(|m| m.labels.clone())
            .unwrap_or_else(Vec::new);
        let mut labeled_features = get_features_labeled(&samples, &mut labels, true)?;
        let mut test_labeled_features = get_features_labeled(&test_samples, &mut labels, false)?;
        println!("Train samples {}.", labeled_features.len());
        println!("Test samples {}.", test_labeled_features.len());
        println!("Labels: {:?}.", labels);
        // validate labels
        if labels.len() < 2 {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                "Your training data need to contain at least two labels",
            ));
        }
        let input_len = wakeword_model
            .as_ref()
            .map(|m| m.train_size * FEATURE_EXTRACTOR_NUM_FEATURES) // force same length
            .unwrap_or_else(|| labeled_features.iter().map(|f| f.0.len()).max().unwrap());
        println!(
            "Training on frames of {}ms",
            (input_len / FEATURE_EXTRACTOR_NUM_FEATURES) * 10
        );
        // pad or truncate data
        labeled_features
            .iter_mut()
            .for_each(|i| i.0.resize(input_len, 0.));
        // pad or truncate test data
        test_labeled_features
            .iter_mut()
            .for_each(|i| i.0.resize(input_len, 0.));
        let dataset = WakewordDataset {
            features: input_len,
            labels: labels.len(),
            train_features: get_features_tensor(&labeled_features)?,
            train_labels: get_labels_tensor(labeled_features)?,
            test_features: get_features_tensor(&test_labeled_features)?,
            test_labels: get_labels_tensor(test_labeled_features)?,
        };
        let training_args = TrainingArgs {
            learning_rate,
            epochs,
        };
        let var_tensors =
            training_loop::<Mlp>(dataset, &training_args, wakeword_model).map_err(convert_error)?;
        Ok(WakewordModel {
            labels,
            train_size: input_len / FEATURE_EXTRACTOR_NUM_FEATURES,
            model_weights: var_tensors,
        })
    }
    fn train_from_sample_dirs(
        train_dir: String,
        test_dir: String,
        learning_rate: f64,
        epochs: usize,
        wakeword_model: Option<WakewordModel>,
    ) -> Result<WakewordModel, Error> {
        Self::train_from_sample_buffers(
            get_files_data_map(train_dir)?,
            get_files_data_map(test_dir)?,
            learning_rate,
            epochs,
            wakeword_model,
        )
    }
}

fn training_loop<M: ModelImpl>(
    m: WakewordDataset,
    args: &TrainingArgs,
    wakeword: Option<WakewordModel>,
) -> candle_core::Result<ModelWeights> {
    let dev = Device::Cpu;
    let train_labels = m.train_labels;
    let train_features = m.train_features.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let varmap = VarMap::new();
    let from_wakeword = wakeword.is_some();
    let model = new_model_impl::<Mlp>(&varmap, &dev, m.features, m.labels, wakeword.as_ref())?;
    let sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate);
    let test_features = m.test_features.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    if from_wakeword {
        // test current accuracy
        test_model(&model, &test_features, &test_labels, 0., 0)?;
    }
    for epoch in 1..args.epochs {
        let logits = model.forward(&train_features)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;
        // test progress
        test_model(
            &model,
            &test_features,
            &test_labels,
            loss.to_scalar::<f32>()?,
            epoch,
        )?;
    }
    let var_tensors = get_tensors_data(varmap);
    //let tensor = ;
    // VarBuilder::from_tensors(ts, dtype, device);
    Ok(var_tensors)
}

struct WakewordDataset {
    pub train_features: Tensor,
    pub train_labels: Tensor,
    pub test_features: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
    pub features: usize,
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

fn get_files_data_map(train_dir: String) -> Result<HashMap<String, Vec<u8>>, Error> {
    let paths = fs::read_dir(train_dir)?;
    let mut files_map: HashMap<String, Vec<u8>> = HashMap::new();
    for path_result in paths {
        let path = path_result.unwrap().path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        if filename.to_string().ends_with(".wav") {
            files_map.insert(String::from(filename), std::fs::read(path.as_path())?);
        }
    }
    Ok(files_map)
}

fn test_model<M: ModelImpl, S: candle_core::WithDType>(
    model: &M,
    test_features: &Tensor,
    test_labels: &Tensor,
    loss: S,
    epoch: usize,
) -> Result<(), candle_core::Error> {
    let test_logits = model.forward(test_features)?;
    let sum_ok = test_logits
        .argmax(D::Minus1)?
        .eq(test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    println!(
        "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
        loss,
        100. * test_accuracy
    );
    Ok(())
}

fn get_labels_tensor(labeled_features: Vec<(Vec<f32>, u32)>) -> Result<Tensor, Error> {
    Tensor::from_iter(labeled_features.iter().map(|lf| lf.1), &Device::Cpu).map_err(convert_error)
}

fn get_features_tensor(labeled_features: &Vec<(Vec<f32>, u32)>) -> Result<Tensor, Error> {
    let tensors_result: Result<Vec<Tensor>, Error> = labeled_features
        .iter()
        .map(|lf| {
            // TODO: no clone
            Tensor::from_iter(lf.0.iter().cloned(), &Device::Cpu).map_err(convert_error)
        })
        .collect();
    Tensor::stack(&tensors_result?, 0).map_err(convert_error)
}
fn convert_error(err: candle_core::Error) -> Error {
    Error::new(ErrorKind::Other, format!("{}", err))
}
fn get_features_labeled(
    samples: &HashMap<String, Vec<u8>>,
    labels: &mut Vec<String>,
    new_labels: bool,
) -> Result<Vec<(Vec<f32>, u32)>, Error> {
    let mut labeled_data: Vec<(Vec<f32>, u32)> = Vec::new();
    let mut sample_rms_level = 0.;
    for (name, buffer) in samples {
        let init_token_index = name.chars().position(|c| c == '[');
        let end_token_index = name.chars().position(|c| c == ']');
        let label = if init_token_index.is_some()
            && end_token_index.is_some()
            && init_token_index.unwrap() < end_token_index.unwrap()
        {
            name[(init_token_index.unwrap() + 1)..end_token_index.unwrap()].to_lowercase()
        } else {
            "none".to_string()
        };
        if !labels.contains(&label) {
            if new_labels {
                labels.push(label.clone());
            } else {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Label '{}' do not exists on training data", label),
                ));
            }
        }
        let label_index = labels
            .iter()
            .position(|r| r.eq_ignore_ascii_case(&label))
            .unwrap() as u32;
        let features = crate::wakeword::compute_sample_features(
            BufReader::new(buffer.as_slice()),
            &mut sample_rms_level,
        )
        .map_err(|msg| Error::new(ErrorKind::Other, msg))?;
        let flatten_features = flat_features(features);
        labeled_data.push((flatten_features, label_index));
    }
    Ok(labeled_data)
}
