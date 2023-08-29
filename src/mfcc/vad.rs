use crate::config::VADMode;


pub struct VadDetector {
    mode: VADMode,
    // state
    index: usize,
    window: Vec<f32>,
    voice_countdown: usize,
}
impl VadDetector {
    pub fn is_voice(&mut self, mfcc: &[f32]) -> bool {
        let value: f32 = mfcc.iter().map(|v| v * v).sum::<f32>() / mfcc.len() as f32;
        self.window[self.index] = value;
        self.index = if self.index >= self.window.len() - 1 {
            0
        } else {
            self.index + 1
        };
        let min = self
            .window
            .iter()
            .filter(|i| !i.is_nan())
            .min_by(|a, b| a.total_cmp(b))
            .unwrap();
        let th = min * self.mode.get_value();
        let n_high_frames = self.window.iter().filter(|v| **v > th).count();
        if n_high_frames > 10 {
            self.voice_countdown = 100;
        }
        if self.voice_countdown > 0 {
            self.voice_countdown -= 1;
            true
        } else {
            false
        }
    }
    pub fn reset(&mut self) {
        self.window.fill(f32::NAN);
        self.voice_countdown = 0;
        self.index = 0;
    }
    pub fn new(mode: VADMode) -> VadDetector {
        VadDetector {
            index: 0,
            window: vec![f32::NAN; 50],
            voice_countdown: 0,
            mode
        }
    }
}
