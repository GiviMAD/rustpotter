use crate::{RustpotterDetection, ScoreMode};

pub(crate) trait WakewordDetector: Send {
    fn get_mfcc_frame_size(&self) -> usize;
    fn get_mfcc_size(&self) -> u16;
    fn run_detection(
        &self,
        mfcc_frame: Vec<Vec<f32>>,
        avg_threshold: f32,
        threshold: f32,
    ) -> Option<RustpotterDetection>;
    fn get_rms_level(&self) -> f32;
    fn update_config(&mut self, score_ref: f32, band_size: u16, score_mode: ScoreMode);
}
