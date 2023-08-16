use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::nn::{WakewordModelTrain, WakewordNN};

use super::{DeserializableWakeword, SerializableWakeword};
#[derive(Serialize, Deserialize)]
pub struct WakewordModel {
    pub labels: Vec<String>,
    pub train_size: usize,
    pub m_type: WakewordModelType,
    pub weights: ModelWeights,
    pub rms_level: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum WakewordModelType {
    SMALL,
    MEDIUM,
    LARGE,
}

impl WakewordModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SMALL => "small",
            Self::MEDIUM => "medium",
            Self::LARGE => "large"
        }
    }
}

impl std::str::FromStr for WakewordModelType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "small" => Ok(Self::SMALL),
            "medium" => Ok(Self::MEDIUM),
            "large" => Ok(Self::LARGE),
            _ => Err("Unknown model type". to_string()),
        }
    }
}
impl SerializableWakeword for WakewordModel {}
impl DeserializableWakeword for WakewordModel {}

impl WakewordModel {
    pub(crate) fn get_nn(&self, score_ref: f32) -> WakewordNN {
        WakewordNN::new(self, score_ref)
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
