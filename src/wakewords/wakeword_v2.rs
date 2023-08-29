use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{WakewordLoad, WakewordRef, WakewordSave};

/// Wakeword representation.
#[derive(Serialize, Deserialize)]
pub struct WakewordV2 {
    pub name: String,
    pub avg_features: Option<Vec<Vec<f32>>>,
    pub samples_features: HashMap<String, Vec<Vec<f32>>>,
    pub threshold: Option<f32>,
    pub avg_threshold: Option<f32>,
    pub rms_level: f32,
    pub enabled: bool,
}
impl WakewordLoad for WakewordV2 {}
impl WakewordSave for WakewordV2 {}
impl Into<WakewordRef> for WakewordV2 {
    fn into(self) -> WakewordRef {
        let mfcc_size = self.samples_features.values().next().unwrap()[0].len() as u16;
        WakewordRef {
            name: self.name,
            avg_features: self.avg_features,
            samples_features: self.samples_features,
            threshold: self.threshold,
            avg_threshold: self.avg_threshold,
            rms_level: self.rms_level,
            mfcc_size,
        }
    }
}
