pub(crate) struct GainNormalizerFilter {
    pub target_rms_level: f32,
}
impl GainNormalizerFilter {
    pub fn filter(&self, signal: &mut [f32], rms_level: f32) -> f32 {
        if self.target_rms_level != f32::NAN && rms_level != 0. {
            let mut gain = self.target_rms_level / rms_level;
            // only apply to frames that are at least twice over or at least half the reference level
            if gain <= 0.5 || (1.5 < gain && gain <= 2.) {
                gain = (gain * 10.).round() / 10.;
                for sample in signal {
                    *sample *= gain;
                }
                return gain;
            }
        }
        1.
    }
    pub fn get_rms_level(signal: &[f32]) -> f32 {
        let mut sum_squared = 0.0;
        for sample in signal {
            sum_squared += sample * sample;
        }
        (sum_squared / signal.len() as f32).sqrt()
    }
    pub fn new() -> GainNormalizerFilter {
        GainNormalizerFilter {
            target_rms_level: f32::NAN,
        }
    }
}
