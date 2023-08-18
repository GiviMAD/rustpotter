use crate::constants::MFCCS_EXTRACTOR_OUT_BANDS;

use super::Dtw;
use std::cmp;

pub struct MfccComparator {
    score_ref: f32,
}
impl MfccComparator {
    pub fn new(score_ref: f32) -> Self {
        MfccComparator {
            score_ref,
        }
    }
    pub fn calculate_distance(ax: &[f32], bx: &[f32]) -> f32 {
        1. - cosine_similarity(ax, bx)
    }
    pub fn compare(&self, a: &[&[f32]], b: &[&[f32]]) -> f32 {
        let mut dtw = Dtw::new(MfccComparator::calculate_distance);
        let cost = dtw.compute_optimal_path_with_window(a, b, MFCCS_EXTRACTOR_OUT_BANDS as u16);
        let normalized_cost = cost / (a.len() + b.len()) as f32;
        self.compute_probability(normalized_cost)
    }
    fn compute_probability(&self, cost: f32) -> f32 {
        1. / (1. + ((cost - self.score_ref) / self.score_ref).exp())
    }
}
impl Clone for MfccComparator {
    fn clone(&self) -> Self {
       Self::new(self.score_ref)
    }
}
pub fn cosine_similarity(vector_a: &[f32], vector_b: &[f32]) -> f32 {
    let dimensionality = cmp::min(vector_a.len(), vector_b.len());
    let mut dot_ab = 0.;
    let mut dot_a = 0.;
    let mut dot_b = 0.;
    let mut dimension = 0;
    while dimension < dimensionality {
        let component_a = vector_a[dimension];
        let component_b = vector_b[dimension];
        dot_ab += component_a * component_b;
        dot_a += component_a * component_a;
        dot_b += component_b * component_b;
        dimension += 1;
    }
    let magnitude = f32::sqrt(dot_a * dot_b);
    if magnitude == 0. {
        0.
    } else {
        dot_ab / magnitude
    }
}
