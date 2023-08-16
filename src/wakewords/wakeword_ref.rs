use super::comp::WakewordComparator;
use crate::{
    mfcc::MfccComparator,
    wakewords::{WakewordSave, WakewordLoad},
    ScoreMode, WakewordRefBuildFromBuffers, WakewordRefBuildFromFiles,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wakeword representation.
#[derive(Serialize, Deserialize)]
pub struct WakewordRef {
    pub name: String,
    pub avg_features: Option<Vec<Vec<f32>>>,
    pub samples_features: HashMap<String, Vec<Vec<f32>>>,
    pub threshold: Option<f32>,
    pub avg_threshold: Option<f32>,
    pub rms_level: f32,
    pub enabled: bool,
}
impl WakewordLoad for WakewordRef {}
impl WakewordSave for WakewordRef {}
impl WakewordRefBuildFromBuffers for WakewordRef {}
impl WakewordRefBuildFromFiles for WakewordRef {}
impl WakewordRef {
    pub(crate) fn new(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        avg_features: Option<Vec<Vec<f32>>>,
        rms_level: f32,
        samples_features: HashMap<String, Vec<Vec<f32>>>,
    ) -> Result<WakewordRef, String> {
        if samples_features.is_empty() {
            return Err("Can not create an empty wakeword".to_string());
        }
        Ok(WakewordRef {
            name,
            threshold,
            avg_threshold,
            avg_features,
            samples_features,
            rms_level,
            enabled: true,
        })
    }
    pub(crate) fn get_comparator(
        &self,
        feature_comparator: MfccComparator,
        score_mode: ScoreMode,
    ) -> WakewordComparator {
        WakewordComparator::new(self, feature_comparator, score_mode)
    }
}
