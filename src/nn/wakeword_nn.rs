use crate::{
    constants::{FEATURE_EXTRACTOR_NUM_FEATURES, NN_NONE_LABEL},
    wakeword_model::{ModelWeights, TensorData},
    RustpotterDetection, WakewordModel,
};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, VarBuilder, VarMap};
use std::{collections::HashMap, io::Cursor, str::FromStr};

pub(crate) struct WakewordNN {
    _vars: VarMap,
    model: Mlp,
    features_size: usize,
    labels: Vec<String>,
}

impl WakewordNN {
    pub fn new(wakeword_model: &WakewordModel) -> Self {
        let vars = VarMap::new();
        let model = new_model_impl::<Mlp>(
            &vars,
            &Device::Cpu,
            wakeword_model.train_size * FEATURE_EXTRACTOR_NUM_FEATURES,
            wakeword_model.labels.len(),
            Some(wakeword_model),
        )
        .unwrap();
        WakewordNN {
            _vars: vars,
            model,
            labels: wakeword_model.labels.clone(),
            features_size: wakeword_model.train_size,
        }
    }
    pub fn contains_label(&self, label: &str) -> bool {
        self.labels.contains(&label.to_string())
    }
    pub fn get_required_features(&self) -> usize {
        self.features_size
    }
    pub fn run_detection(&self, features: Vec<Vec<f32>>) -> Option<RustpotterDetection> {
        assert!(features.len() == self.features_size);
        Tensor::from_iter(flat_features(features).into_iter(), &Device::Cpu)
            .and_then(|tensor| Tensor::stack(&[tensor], 0))
            .and_then(|tensor_stack| self.model.forward(&tensor_stack))
            .and_then(|logits| logits.get(0))
            .and_then(|logits1| logits1.to_vec1::<f32>())
            .map_err(|err| {
                println!("Error running wakeword nn: {}", err);
                err
            })
            .ok()
            .map(|prob_vec| {
                self.get_label(&prob_vec)
                    .map(|label| self.run_detection_by_label(&label, &prob_vec))
                    .unwrap_or(None)
            })
            .unwrap_or(None)
    }

    fn get_label(&self, prob_vec: &[f32]) -> Option<&str> {
        prob_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .map(|i| {
                if i < self.labels.len() {
                    &self.labels[i]
                } else {
                    NN_NONE_LABEL
                }
            })
    }
    fn run_detection_by_label(&self, label: &str, prob_vec: &[f32]) -> Option<RustpotterDetection> {
        if !NN_NONE_LABEL.eq_ignore_ascii_case(&label) {
            let scores: HashMap<String, f32> = prob_vec
                .iter()
                .enumerate()
                .map(|(i, prob)| (self.labels[i].clone(), prob.clone()))
                .collect();
            let none_prob = scores.get(NN_NONE_LABEL).unwrap_or(&0.);
            let label_prob = scores.get(label).unwrap_or(&0.);
            let second_prob = prob_vec
                .into_iter()
                .filter(|p| !label_prob.eq(*p))
                .max_by(|a, b| b.total_cmp(a))
                .unwrap_or(&0.);
            Some(RustpotterDetection {
                name: label.to_string(),
                avg_score: calc_inverse_similarity(label_prob, second_prob),
                score: calc_inverse_similarity(label_prob, none_prob),
                scores,
                counter: usize::MIN, // added by the detector
                gain: f32::NAN,      // added by the detector
            })
        } else {
            None
        }
    }
}

fn calc_inverse_similarity(n1: &f32, n2: &f32) -> f32 {
    ((n1 - n2) / (n1 + n2).abs()).min(1.)
}
pub(crate) fn new_model_impl<M: ModelImpl>(
    varmap: &VarMap,
    dev: &Device,
    features_size: usize,
    labels_size: usize,
    wakeword: Option<&WakewordModel>,
) -> Result<M, candle_core::Error> {
    let vs = VarBuilder::from_varmap(varmap, DType::F32, dev);
    let model = M::new(vs.clone(), features_size, labels_size)?;
    if let Some(wakeword) = wakeword {
        load_varmap(varmap, &wakeword.model_weights)?;
    }
    Ok(model)
}
pub(crate) fn get_tensors_data(varmap: VarMap) -> ModelWeights {
    let mut model_weights: HashMap<String, TensorData> = HashMap::new();
    for (name, tensor) in varmap.data().lock().unwrap().iter() {
        model_weights.insert(name.to_string(), tensor.into());
    }
    model_weights
}

impl From<&Var> for TensorData {
    fn from(tensor: &Var) -> Self {
        let mut w = Cursor::new(Vec::new());
        let dims = tensor.shape().clone().into_dims();
        tensor.write_bytes(&mut w).unwrap();
        TensorData {
            bytes: w.into_inner(),
            d_type: tensor.dtype().as_str().to_string(),
            dims,
        }
    }
}
fn load_varmap(
    varmap: &VarMap,
    model_weights: &HashMap<String, TensorData>,
) -> Result<(), candle_core::Error> {
    for (name, var) in varmap.data().lock().unwrap().iter_mut() {
        match model_weights.get(name) {
            Some(data) => {
                var.set(&data.into()).expect("Error loading model varmap");
            }
            None => panic!("Incorrect model layers"),
        };
    }
    Ok(())
}

impl Into<Tensor> for &TensorData {
    fn into(self) -> Tensor {
        let d_type = DType::from_str(&self.d_type).unwrap_or(DType::F32);
        Tensor::from_raw_buffer(&self.bytes, d_type, &self.dims, &Device::Cpu).unwrap()
    }
}

pub(crate) fn flat_features(features: Vec<Vec<f32>>) -> Vec<f32> {
    features
        .into_iter()
        .flat_map(|array| array.into_iter())
        .collect()
}
pub(crate) struct Mlp {
    ln1: Linear,
    ln2: Linear,
}
struct LinearModel {
    linear: Linear,
}
fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> candle_core::Result<Linear> {
    let ws = vs.get_or_init((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_or_init(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}
pub(crate) trait ModelImpl: Sized {
    fn new(vs: VarBuilder, input_size: usize, labels_size: usize) -> candle_core::Result<Self>;
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor>;
}

impl ModelImpl for LinearModel {
    fn new(vs: VarBuilder, input_size: usize, labels_size: usize) -> candle_core::Result<Self> {
        let linear = linear_z(input_size, labels_size, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.linear.forward(xs)
    }
}

impl ModelImpl for Mlp {
    fn new(vs: VarBuilder, input_size: usize, labels_size: usize) -> candle_core::Result<Self> {
        let inter_size = input_size / 3;
        let ln1 = candle_nn::linear(input_size, inter_size, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(inter_size, labels_size, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}
