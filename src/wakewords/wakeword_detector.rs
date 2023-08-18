use crate::RustpotterDetection;

pub(crate) trait WakewordDetector: Send {
    fn get_mfcc_frame_size(&self) -> usize;
    fn run_detection(&self, mfcc_frame: Vec<Vec<f32>>, avg_threshold: f32, threshold: f32) -> Option<RustpotterDetection>;
    fn contains(&self, name: &str) -> bool;
    fn get_rms_level(&self) -> f32;
}
