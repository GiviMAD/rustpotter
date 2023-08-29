use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ScoreMode;

use super::nn::{WakewordModelTrain, WakewordNN};

use super::{WakewordSave, WakewordLoad, WakewordFile, WakewordDetector};
#[derive(Serialize, Deserialize)]
pub struct WakewordModel {
    pub labels: Vec<String>,
    pub train_size: usize,
    pub mfcc_size: u16,
    pub m_type: ModelType,
    pub weights: ModelWeights,
    pub rms_level: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum ModelType {
    SMALL,
    MEDIUM,
    LARGE,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SMALL => "small",
            Self::MEDIUM => "medium",
            Self::LARGE => "large"
        }
    }
}

impl std::str::FromStr for ModelType {
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
impl WakewordLoad for WakewordModel {}
impl WakewordSave for WakewordModel {}
impl WakewordFile for WakewordModel {
    fn get_detector(&self, score_ref: f32, _: u16, _: ScoreMode) -> Box<dyn WakewordDetector> {
        Box::new(WakewordNN::new(self, score_ref))
    }

    fn get_mfcc_size(&self) -> u16 {
        self.mfcc_size
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
