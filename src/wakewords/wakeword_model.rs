use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ScoreMode;

use super::nn::{WakewordModelTrain, WakewordNN};

use super::{WakewordDetector, WakewordFile, WakewordLoad, WakewordSave};
#[derive(Serialize, Deserialize)]
pub struct WakewordModel {
    pub labels: Vec<String>,
    pub train_size: usize,
    pub mfcc_size: u16,
    pub m_type: ModelType,
    pub weights: ModelWeights,
    pub rms_level: f32,
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

#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Serialize, Deserialize, Clone)]
pub enum ModelType {
    Tiny = 0,
    Small = 1,
    Medium = 2,
    Large = 3,
}
impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}
impl std::str::FromStr for ModelType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err("Unknown model type".to_string()),
        }
    }
}
#[cfg(feature = "display")]
impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
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
