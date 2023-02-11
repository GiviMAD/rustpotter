pub(crate) struct GainNormalizerFilter {
    pub target_rms_level: f32,
}
impl GainNormalizerFilter {
    pub fn filter(&self, signal: &mut [f32]) {
        if self.target_rms_level != f32::NAN {
            let rms = self.get_rms_level(signal);
            if rms != 0. {
                let mut gain = self.target_rms_level / rms;
                // only apply to samples that are at least twice over the reference level
                if gain <= 0.5 {
                    gain = (gain * 10.).round() / 10.;
                    for sample in signal {
                        *sample *= gain;
                    }
                }
            }
        }
    }
    pub fn get_rms_level(&self, signal: &[f32]) -> f32 {
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
