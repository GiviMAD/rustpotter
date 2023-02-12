use super::Dtw;
use std::cmp;

pub struct FeatureComparator {
    band_size: u16,
    reference: f32,
}
impl FeatureComparator {
    pub fn compare(&self, a: &[&[f32]], b: &[&[f32]]) -> f32 {
        let mut dtw = Dtw::new(FeatureComparator::calculate_distance);
        let cost = dtw.compute_optimal_path_with_window(a, b, self.band_size);
        let normalized_cost = cost / (a.len() + b.len()) as f32;
        self.compute_probability(normalized_cost)
    }
    fn compute_probability(&self, cost: f32) -> f32 {
        1. / (1. + ((cost - self.reference) / self.reference).exp())
    }
    pub fn new(band_size: u16, reference: f32) -> Self {
        FeatureComparator {
            band_size,
            reference,
        }
    }
    pub fn calculate_distance(ax: &[f32], bx: &[f32]) -> f32 {
        1. - cosine_similarity(ax, bx)
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
