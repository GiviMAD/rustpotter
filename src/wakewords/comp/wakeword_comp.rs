use std::collections::HashMap;

use crate::{
    mfcc::{MfccComparator, MfccNormalizer},
    wakewords::WakewordDetector,
    RustpotterDetection, ScoreMode, WakewordRef,
};

pub(crate) struct WakewordComparator {
    name: String,
    avg_features: Option<Vec<Vec<f32>>>,
    samples_features: HashMap<String, Vec<Vec<f32>>>,
    threshold: Option<f32>,
    avg_threshold: Option<f32>,
    rms_level: f32,
    // state
    score_mode: ScoreMode,
    mfcc_comparator: MfccComparator,
}

impl WakewordComparator {
    fn cut_and_normalize_frame(
        &self,
        mut mfccs: Vec<Vec<f32>>,
        max_len: usize,
    ) -> Vec<Vec<f32>> {
        if mfccs.len() > max_len {
            mfccs.drain(max_len..mfccs.len());
        }
        MfccNormalizer::normalize(mfccs)
    }
    fn score_frame(&self, frame_features: &[Vec<f32>], template: &[Vec<f32>]) -> f32 {
        let score = self.mfcc_comparator.compare(
            &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            &frame_features
                .iter()
                .map(|item| &item[..])
                .collect::<Vec<_>>(),
        );
        score
    }
    fn get_percentile(&self, sorted_values: &[f32], percentile: f32) -> f32 {
        let n = sorted_values.len();
        let index = percentile / 100.0 * (n - 1) as f32;
        let index_floor = index.floor();
        if index_floor == index {
            sorted_values[index as usize]
        } else {
            let i = index_floor as usize;
            let d = index - index_floor;
            sorted_values[i] * (1.0 - d) + sorted_values[i + 1] * d
        }
    }
    pub fn new(
        wakeword: &WakewordRef,
        mfcc_comparator: MfccComparator,
        score_mode: ScoreMode,
    ) -> Self {
        WakewordComparator {
            name: wakeword.name.clone(),
            avg_features: wakeword.avg_features.clone(),
            samples_features: wakeword.samples_features.clone(),
            threshold: wakeword.threshold,
            avg_threshold: wakeword.avg_threshold,
            rms_level: wakeword.rms_level,
            score_mode,
            mfcc_comparator,
        }
    }
}

impl WakewordDetector for WakewordComparator {
    fn get_mfcc_frame_size(&self) -> usize {
        self.samples_features
            .values()
            .map(Vec::len)
            .max()
            .unwrap_or(usize::MIN)
    }

    fn run_detection(
        &self,
        mfcc_frame: Vec<Vec<f32>>,
        avg_threshold: f32,
        threshold: f32,
    ) -> Option<RustpotterDetection> {
        let avg_threshold = self.avg_threshold.unwrap_or(avg_threshold);
        let mut avg_score = 0.;
        if self.avg_features.is_some() && avg_threshold != 0. {
            // discard detections against the wakeword averaged features
            let wakeword_samples_avg_features = self.avg_features.as_ref().unwrap();
            let mfcc_window_normalized = self
                .cut_and_normalize_frame(mfcc_frame.to_vec(), wakeword_samples_avg_features.len());
            avg_score = self.score_frame(&mfcc_window_normalized, wakeword_samples_avg_features);
            if avg_score < avg_threshold {
                return None;
            }
        }
        let threshold = self.threshold.unwrap_or(threshold);
        let scores = self.samples_features.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, f32>, (name, wakeword_sample_features)| {
                let frame_features_normalized = self
                    .cut_and_normalize_frame(mfcc_frame.to_vec(), wakeword_sample_features.len());
                acc.insert(
                    name.to_string(),
                    self.score_frame(&frame_features_normalized, wakeword_sample_features),
                );
                acc
            },
        );
        let mut sorted_scores = scores.values().copied().collect::<Vec<f32>>();
        let score = match self.score_mode {
            ScoreMode::Average => sorted_scores.iter().sum::<f32>() / sorted_scores.len() as f32,
            ScoreMode::Max => {
                sorted_scores.sort_by(|a, b| b.total_cmp(a));
                sorted_scores[0]
            }
            ScoreMode::Median | ScoreMode::P50 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 50.)
            }
            ScoreMode::P25 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 25.)
            }
            ScoreMode::P75 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 75.)
            }
            ScoreMode::P80 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 80.)
            }
            ScoreMode::P90 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 90.)
            }
            ScoreMode::P95 => {
                sorted_scores.sort_by(|a, b| a.total_cmp(b));
                self.get_percentile(&sorted_scores, 95.)
            }
        };
        if score > threshold {
            Some(RustpotterDetection {
                name: self.name.clone(),
                avg_score,
                score,
                scores,
                counter: usize::MIN, // added by the detector
                gain: f32::NAN,      // added by the detector
            })
        } else {
            None
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.name.eq(name)
    }

    fn get_rms_level(&self) -> f32 {
        self.rms_level
    }

    fn get_mfcc_size(&self) -> u16 {
        self.samples_features.values().next().unwrap()[0].len() as u16
    }
}
