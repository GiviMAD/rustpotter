pub(crate) struct GainNormalizerFilter {
    window_size: usize,
    fixed_rms_level: bool,
    min_gain: f32,
    max_gain: f32,
    // state
    rms_level_ref: f32,
    rms_level_window: Vec<f32>,
}
impl GainNormalizerFilter {
    pub fn filter(&mut self, signal: &mut [f32], rms_level: f32) -> f32 {
        if !self.rms_level_ref.is_nan() && rms_level != 0. {
            // update the window
            self.rms_level_window.push(rms_level);
            if self.rms_level_window.len() > self.window_size {
                self.rms_level_window.drain(0..1);
            }
            // calculate the mean of the rms window
            let frame_rms_level =
                self.rms_level_window.iter().sum::<f32>() / self.rms_level_window.len() as f32;
            // calculate the gain to apply
            let mut gain = self.rms_level_ref / frame_rms_level;
            // range and round the value, trying to get a mostly uniform gain between frames.
            gain = ((gain * 10.).round() / 10.).clamp(self.min_gain, self.max_gain);
            // apply gain unless irrelevant
            if gain != 1. {
                for sample in signal {
                    *sample = (*sample * gain).clamp(-1., 1.);
                }
            }
            gain
        } else {
            1.
        }
    }
    pub fn get_rms_level_ref(&self) -> f32 {
        self.rms_level_ref
    }
    pub fn set_rms_level_ref(&mut self, rms_level: f32, window_size: usize) {
        if !self.fixed_rms_level {
            self.rms_level_ref = rms_level;
        }
        self.window_size = if window_size != 0 { window_size } else { 1 };
    }
    pub fn get_rms_level(signal: &[f32]) -> f32 {
        let mut sum_squared = 0.0;
        for sample in signal {
            sum_squared += sample * sample;
        }
        (sum_squared / signal.len() as f32).sqrt()
    }
    pub fn new(min_gain: f32, max_gain: f32, fixed_rms_level: Option<f32>) -> GainNormalizerFilter {
        GainNormalizerFilter {
            min_gain,
            max_gain,
            rms_level_ref: fixed_rms_level.unwrap_or(f32::NAN),
            fixed_rms_level: fixed_rms_level.is_some(),
            rms_level_window: Vec::new(),
            window_size: 1,
        }
    }
}
