use super::wakeword_nn::{
    flat_features, get_tensors_data, init_model_impl, LargeModel, MediumModel, ModelImpl, TinyModel,
};
use crate::{
    constants::NN_NONE_LABEL,
    mfcc::MfccWavFileExtractor,
    wakewords::{nn::wakeword_nn::SmallModel, ModelWeights},
    ModelType, WakewordModel,
};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, VarMap};
use std::{
    collections::HashMap,
    fs,
    io::{BufReader, Error, ErrorKind},
};

pub struct WakewordModelTrainOptions {
    m_type: ModelType,
    learning_rate: f64,
    epochs: usize,
    test_epochs: usize,
    mfcc_size: u16,
}

impl WakewordModelTrainOptions {
    pub fn new(
        m_type: ModelType,
        learning_rate: f64,
        epochs: usize,
        test_epochs: usize,
        mfcc_size: u16,
    ) -> Self {
        Self {
            m_type,
            learning_rate,
            epochs,
            test_epochs,
            mfcc_size,
        }
    }
}

pub trait WakewordModelTrain {
    fn train_from_buffers(
        config: WakewordModelTrainOptions,
        samples: HashMap<String, Vec<u8>>,
        test_samples: HashMap<String, Vec<u8>>,
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
        // prepare data
        if wakeword_model.is_some() {
            println!("Training from previous model, some options will be ignored.");
        }
        let mut labels: Vec<String> = wakeword_model
            .as_ref()
            .map(|m| m.labels.clone())
            .unwrap_or_else(Vec::new);
        let m_type = wakeword_model
            .as_ref()
            .map(|m| m.m_type.clone())
            .unwrap_or(config.m_type);
        let mfcc_size = wakeword_model
            .as_ref()
            .map(|m| m.mfcc_size)
            .unwrap_or(config.mfcc_size);
        let mut rms_level: f32 = f32::NAN;
        let mut labeled_mfccs = get_mfccs_labeled(
            &samples,
            &mut labels,
            &mut rms_level,
            wakeword_model.is_none(),
            mfcc_size,
        )?;
        let mut noop_rms_level: f32 = f32::NAN;
        let mut test_labeled_mfccs = get_mfccs_labeled(
            &test_samples,
            &mut labels,
            &mut noop_rms_level,
            false,
            mfcc_size,
        )?;
        println!("Model type: {}.", m_type.as_str());
        println!("Labels: {:?}.", labels);
        println!("Training with {} records.", labeled_mfccs.len());
        println!("Testing with {} records.", test_labeled_mfccs.len());
        // validate labels
        if labels.len() < 2 {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                "Your training data need to contain at least two labels",
            ));
        }
        // use previous training size or get it from the training samples
        let input_len = wakeword_model
            .as_ref()
            .map(|m| m.train_size * mfcc_size as usize)
            .unwrap_or_else(|| labeled_mfccs.iter().map(|f| f.0.len()).max().unwrap());
        println!(
            "Training on {}ms of audio",
            (input_len / mfcc_size as usize) * 10
        );
        // pad or truncate data
        labeled_mfccs
            .iter_mut()
            .chain(test_labeled_mfccs.iter_mut())
            .for_each(|i| i.0.resize(input_len, 0.));
        // run training loop
        let dataset = WakewordDataset {
            input_len,
            mfcc_size,
            labels: labels.len(),
            train_labels: get_labels_tensor_stack(&labeled_mfccs)?,
            train_features: get_mfccs_tensor_stack(labeled_mfccs)?,
            test_labels: get_labels_tensor_stack(&test_labeled_mfccs)?,
            test_features: get_mfccs_tensor_stack(test_labeled_mfccs)?,
        };
        let training_args = TrainingArgs {
            learning_rate: config.learning_rate,
            epochs: config.epochs,
            test_epochs: config.test_epochs,
        };
        let weights = match m_type {
            ModelType::Tiny => training_loop::<TinyModel>(dataset, &training_args, wakeword_model)
                .map_err(convert_error)?,
            ModelType::Small => {
                training_loop::<SmallModel>(dataset, &training_args, wakeword_model)
                    .map_err(convert_error)?
            }
            ModelType::Medium => {
                training_loop::<MediumModel>(dataset, &training_args, wakeword_model)
                    .map_err(convert_error)?
            }
            ModelType::Large => {
                training_loop::<LargeModel>(dataset, &training_args, wakeword_model)
                    .map_err(convert_error)?
            }
        };
        // return serializable model struct
        Ok(WakewordModel {
            labels,
            m_type,
            train_size: input_len / mfcc_size as usize,
            mfcc_size,
            weights,
            rms_level,
        })
    }
    fn train_from_dirs(
        config: WakewordModelTrainOptions,
        train_dir: String,
        test_dir: String,
        wakeword_model: Option<WakewordModel>,
    ) -> Result<WakewordModel, Error> {
        Self::train_from_buffers(
            config,
            get_files_data_map(train_dir)?,
            get_files_data_map(test_dir)?,
            wakeword_model,
        )
    }
}

fn training_loop<M: ModelImpl + 'static>(
    m: WakewordDataset,
    args: &TrainingArgs,
    wakeword: Option<WakewordModel>,
) -> candle_core::Result<ModelWeights> {
    let dev = Device::Cpu;
    let train_labels = m.train_labels;
    let train_features = m.train_features.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let var_map = VarMap::new();
    let from_wakeword = wakeword.is_some();
    let model = init_model_impl::<M>(
        &var_map,
        &dev,
        m.input_len,
        m.mfcc_size,
        m.labels,
        wakeword.as_ref(),
    )?;
    let sgd = candle_nn::SGD::new(var_map.all_vars(), args.learning_rate);
    let test_features = m.test_features.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    if from_wakeword {
        // test current accuracy
        test_model(model.as_ref(), &test_features, &test_labels, 0., 0)?;
    }
    for epoch in 1..=args.epochs {
        let logits = model.forward(&train_features)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;
        // test progress
        if (epoch % args.test_epochs) == 0 || epoch == args.epochs {
            test_model(
                model.as_ref(),
                &test_features,
                &test_labels,
                loss.to_scalar::<f32>()?,
                epoch,
            )?;
        }
    }
    Ok(get_tensors_data(var_map))
}

struct WakewordDataset {
    pub train_features: Tensor,
    pub train_labels: Tensor,
    pub test_features: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
    pub input_len: usize,
    pub mfcc_size: u16,
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
    test_epochs: usize,
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

fn test_model<S: candle_core::WithDType>(
    model: &dyn ModelImpl,
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

fn get_labels_tensor_stack(labeled_features: &[(Vec<f32>, u32)]) -> Result<Tensor, Error> {
    Tensor::from_iter(labeled_features.iter().map(|lf| lf.1), &Device::Cpu).map_err(convert_error)
}

fn get_mfccs_tensor_stack(labeled_mfccs: Vec<(Vec<f32>, u32)>) -> Result<Tensor, Error> {
    let tensors_result: Result<Vec<Tensor>, Error> = labeled_mfccs
        .into_iter()
        .map(|lf| Tensor::from_iter(lf.0.into_iter(), &Device::Cpu).map_err(convert_error))
        .collect();
    Tensor::stack(&tensors_result?, 0).map_err(convert_error)
}
fn convert_error(err: candle_core::Error) -> Error {
    Error::new(ErrorKind::Other, format!("{}", err))
}
fn get_mfccs_labeled(
    samples: &HashMap<String, Vec<u8>>,
    labels: &mut Vec<String>,
    sample_rms_level: &mut f32,
    new_labels: bool,
    mfcc_size: u16,
) -> Result<Vec<(Vec<f32>, u32)>, Error> {
    let mut labeled_data: Vec<(Vec<f32>, u32)> = Vec::new();
    for (name, buffer) in samples {
        let init_token_index = name.chars().position(|c| c == '[');
        let end_token_index = name
            .chars()
            .position(|c| c == ']')
            .filter(|end| init_token_index.map(|start| start < *end).unwrap_or(false));
        let label = if let (Some(init_token_index), Some(end_token_index)) =
            (init_token_index, end_token_index)
        {
            name[init_token_index + 1..end_token_index].to_lowercase()
        } else {
            NN_NONE_LABEL.to_string()
        };
        if !labels.contains(&label) {
            if new_labels {
                labels.push(label.clone());
            } else {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Forbidden label '{}', it doesn't exists on the training data or in the model you are training from.", label),
                ));
            }
        }
        let label_index = labels.iter().position(|r| r.eq(&label)).unwrap() as u32;
        let mut tmp_sample_rms_level: f32 = 0.;
        let mfccs: Vec<Vec<f32>> = MfccWavFileExtractor::compute_mfccs(
            BufReader::new(buffer.as_slice()),
            &mut tmp_sample_rms_level,
            mfcc_size,
        )
        .map_err(|msg| Error::new(ErrorKind::Other, msg))?;
        if !label.eq(NN_NONE_LABEL) {
            if !sample_rms_level.is_nan() {
                *sample_rms_level = (*sample_rms_level + tmp_sample_rms_level) / 2.;
            } else {
                *sample_rms_level = tmp_sample_rms_level;
            }
        }
        let flatten_mfccs = flat_features(mfccs);
        labeled_data.push((flatten_mfccs, label_index));
    }
    Ok(labeled_data)
}
