use crate::{
    constants::{MFCCS_EXTRACTOR_OUT_SHIFTS, NN_NONE_LABEL},
    mfcc::MfccNormalizer,
    wakewords::WakewordDetector,
    wakewords::{ModelWeights, TensorData},
    ModelType, RustpotterDetection, WakewordModel,
};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use std::{collections::HashMap, io::Cursor, str::FromStr};

pub(crate) struct WakewordNN {
    _var_map: VarMap,
    model: Box<dyn ModelImpl>,
    mfcc_frames: usize,
    labels: Vec<String>,
    score_ref: f32,
    rms_level: f32,
    mfcc_size: u16,
}

impl WakewordNN {
    pub fn new(wakeword_model: &WakewordModel, score_ref: f32) -> Self {
        let var_map = VarMap::new();
        let m_type = wakeword_model.m_type.clone();
        let model = init_model(
            m_type,
            &var_map,
            &Device::Cpu,
            wakeword_model.train_size * wakeword_model.mfcc_size as usize,
            wakeword_model.mfcc_size,
            wakeword_model.labels.len(),
            Some(wakeword_model),
        )
        .unwrap();
        WakewordNN {
            _var_map: var_map,
            model,
            score_ref,
            rms_level: wakeword_model.rms_level,
            labels: wakeword_model.labels.clone(),
            mfcc_frames: wakeword_model.train_size,
            mfcc_size: wakeword_model.mfcc_size,
        }
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
    fn run_detection_by_label(
        &self,
        label: &str,
        prob_vec: &[f32],
        calc_avg_score: bool,
    ) -> Option<RustpotterDetection> {
        if !NN_NONE_LABEL.eq(label) {
            let scores: HashMap<String, f32> = prob_vec
                .iter()
                .enumerate()
                .map(|(i, prob)| (self.labels[i].clone(), prob.clone()))
                .collect();
            let none_prob = scores.get(NN_NONE_LABEL).unwrap_or(&0.);
            let label_prob = scores.get(label).unwrap_or(&0.);
            let second_prob = if calc_avg_score {
                prob_vec
                    .into_iter()
                    .filter(|p| !label_prob.eq(*p))
                    .max_by(|a, b| b.total_cmp(a))
                    .unwrap_or(&0.)
            } else {
                &0.
            };
            Some(RustpotterDetection {
                name: label.to_string(),
                avg_score: if calc_avg_score {
                    calc_inverse_similarity(label_prob, second_prob, &self.score_ref)
                } else {
                    0.
                },
                score: calc_inverse_similarity(label_prob, none_prob, &self.score_ref),
                scores,
                counter: usize::MIN, // added by the detector
                gain: f32::NAN,      // added by the detector
            })
        } else {
            None
        }
    }

    fn predict_labels(&self, mfcc_frame: Vec<Vec<f32>>) -> Option<Vec<f32>> {
        Tensor::from_iter(flat_features(mfcc_frame).into_iter(), &Device::Cpu)
            .and_then(|tensor| Tensor::stack(&[tensor], 0))
            .and_then(|tensor_stack| self.model.forward(&tensor_stack))
            .and_then(|logits| logits.get(0))
            .and_then(|logits1| logits1.to_vec1::<f32>())
            .map_err(|err| {
                println!("Error running wakeword nn: {}", err);
                err
            })
            .ok()
    }
    fn validate_scores(
        &self,
        detection: RustpotterDetection,
        threshold: f32,
        avg_threshold: f32,
    ) -> Option<RustpotterDetection> {
        if detection.score >= threshold && detection.avg_score >= avg_threshold {
            Some(detection)
        } else {
            None
        }
    }

    fn handle_probabilities(
        &self,
        prob_vec: Vec<f32>,
        calc_avg_score: bool,
    ) -> Option<RustpotterDetection> {
        self.get_label(&prob_vec)
            .and_then(|label| self.run_detection_by_label(&label, &prob_vec, calc_avg_score))
    }
}
impl WakewordDetector for WakewordNN {
    fn get_mfcc_frame_size(&self) -> usize {
        self.mfcc_frames
    }
    fn run_detection(
        &self,
        mut mfcc_frames: Vec<Vec<f32>>,
        avg_threshold: f32,
        threshold: f32,
    ) -> Option<RustpotterDetection> {
        mfcc_frames.truncate(self.mfcc_frames);
        self.predict_labels(MfccNormalizer::normalize(mfcc_frames))
            .and_then(|prob_vec| self.handle_probabilities(prob_vec, avg_threshold != 0.))
            .and_then(|detection| self.validate_scores(detection, threshold, avg_threshold))
    }
    fn contains(&self, name: &str) -> bool {
        self.labels.contains(&name.to_string())
    }
    fn get_rms_level(&self) -> f32 {
        self.rms_level
    }
    fn get_mfcc_size(&self) -> u16 {
        self.mfcc_size
    }
}

fn calc_inverse_similarity(n1: &f32, n2: &f32, reference: &f32) -> f32 {
    let reference = reference * 10.;
    1. - (1. / (1. + (((n1 - n2) - reference) / reference).exp()))
}

pub(crate) fn init_model(
    m_type: ModelType,
    var_map: &VarMap,
    dev: &Device,
    features_size: usize,
    mfcc_size: u16,
    labels_size: usize,
    wakeword: Option<&WakewordModel>,
) -> Result<Box<dyn ModelImpl>, candle_core::Error> {
    match m_type {
        ModelType::SMALL => init_model_impl::<SmallModel>(
            var_map,
            dev,
            features_size,
            mfcc_size,
            labels_size,
            wakeword,
        ),
        ModelType::MEDIUM => init_model_impl::<MediumModel>(
            var_map,
            dev,
            features_size,
            mfcc_size,
            labels_size,
            wakeword,
        ),
        ModelType::LARGE => init_model_impl::<LargeModel>(
            var_map,
            dev,
            features_size,
            mfcc_size,
            labels_size,
            wakeword,
        ),
    }
}

pub(super) fn init_model_impl<M: ModelImpl + 'static>(
    var_map: &VarMap,
    dev: &Device,
    features_size: usize,
    mfcc_size: u16,
    labels_size: usize,
    wakeword: Option<&WakewordModel>,
) -> Result<Box<dyn ModelImpl>, candle_core::Error> {
    let vs = VarBuilder::from_varmap(var_map, DType::F32, dev);
    let model = M::new(vs.clone(), features_size, mfcc_size as usize, labels_size)?;
    if let Some(wakeword) = wakeword {
        load_weights(var_map, &wakeword.weights)?;
    }
    Ok(Box::new(model))
}
pub(crate) fn get_tensors_data(var_map: VarMap) -> ModelWeights {
    let mut model_weights: HashMap<String, TensorData> = HashMap::new();
    for (name, tensor) in var_map.data().lock().unwrap().iter() {
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
fn load_weights(
    var_map: &VarMap,
    model_weights: &HashMap<String, TensorData>,
) -> Result<(), candle_core::Error> {
    for (name, var) in var_map.data().lock().unwrap().iter_mut() {
        let result = model_weights
            .get(name)
            .and_then(|data| Some(var.set(&data.into())))
            .unwrap_or(Err(candle_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Incorrect model layers",
            ))));
        if result.is_err() {
            return result;
        }
    }
    Ok(())
}

impl Into<Tensor> for &TensorData {
    fn into(self) -> Tensor {
        let d_type = DType::from_str(&self.d_type).unwrap_or(DType::F32);
        Tensor::from_raw_buffer(&self.bytes, d_type, &self.dims, &Device::Cpu).unwrap()
    }
}

pub(super) fn flat_features(features: Vec<Vec<f32>>) -> Vec<f32> {
    features
        .into_iter()
        .flat_map(|array| array.into_iter())
        .collect()
}
pub(super) struct LargeModel {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}
pub(super) struct MediumModel {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}
pub(super) struct SmallModel {
    ln1: Linear,
    ln2: Linear,
}
pub(crate) trait ModelImpl: Send {
    fn new(
        vs: VarBuilder,
        input_size: usize,
        mfcc_size: usize,
        labels_size: usize,
    ) -> candle_core::Result<Self>
    where
        Self: Sized;
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor>;
}

impl ModelImpl for SmallModel {
    fn new(
        vs: VarBuilder,
        input_size: usize,
        mfcc_size: usize,
        labels_size: usize,
    ) -> candle_core::Result<Self> {
        let inter_size = (input_size / mfcc_size) / (MFCCS_EXTRACTOR_OUT_SHIFTS * 2);
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

impl ModelImpl for MediumModel {
    fn new(
        vs: VarBuilder,
        input_size: usize,
        mfcc_size: usize,
        labels_size: usize,
    ) -> candle_core::Result<Self> {
        let inter_size1 = (input_size / mfcc_size) / MFCCS_EXTRACTOR_OUT_SHIFTS;
        let ln1 = candle_nn::linear(input_size, inter_size1, vs.pp("ln1"))?;
        let inter_size2 = (input_size / mfcc_size) / (MFCCS_EXTRACTOR_OUT_SHIFTS * 2);
        let ln2: Linear = candle_nn::linear(inter_size1, inter_size2, vs.pp("ln2"))?;
        let ln3: Linear = candle_nn::linear(inter_size2, labels_size, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.ln1.forward(xs)?.relu()?;
        let xs = self.ln2.forward(&xs)?.relu()?;
        self.ln3.forward(&xs)
    }
}

impl ModelImpl for LargeModel {
    fn new(
        vs: VarBuilder,
        input_size: usize,
        mfcc_size: usize,
        labels_size: usize,
    ) -> candle_core::Result<Self> {
        let inter_size1 = ((input_size / mfcc_size) / MFCCS_EXTRACTOR_OUT_SHIFTS) * 2;
        let ln1 = candle_nn::linear(input_size, inter_size1, vs.pp("ln1"))?;
        let inter_size2 = (input_size / mfcc_size) / (MFCCS_EXTRACTOR_OUT_SHIFTS * 2);
        let ln2: Linear = candle_nn::linear(inter_size1, inter_size2, vs.pp("ln2"))?;
        let ln3: Linear = candle_nn::linear(inter_size2, labels_size, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.ln1.forward(xs)?.relu()?;
        let xs = self.ln2.forward(&xs)?.relu()?;
        self.ln3.forward(&xs)
    }
}
