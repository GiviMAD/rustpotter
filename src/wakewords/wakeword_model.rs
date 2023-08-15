use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::nn::{WakewordModelTrain, WakewordNN};

use super::{DeserializableWakeword, SerializableWakeword};
#[derive(Serialize, Deserialize)]
pub struct WakewordModel {
    pub labels: Vec<String>,
    pub train_size: usize,
    pub model_weights: ModelWeights,
    pub rms_level: f32,
}

impl SerializableWakeword for WakewordModel {}
impl DeserializableWakeword for WakewordModel {}

impl WakewordModel {
    pub(crate) fn get_nn(&self) -> WakewordNN {
        WakewordNN::new(self)
    }
}

#[derive(Serialize, Deserialize)]
pub struct TensorData {
    pub bytes: Vec<u8>,
    pub dims: Vec<usize>,
    pub d_type: String,
}
pub type ModelWeights = HashMap<String, TensorData>;

impl WakewordModelTrain for WakewordModel {}
