use crate::config::VADMode;


pub struct VadDetector {
    mode: VADMode,
    // state
    index: usize,
    window: Vec<f32>,
    counter: usize,
}
impl VadDetector {
    pub fn is_voice(&mut self, mfcc: &[f32]) -> bool {
        let value: f32 = mfcc.iter().map(|v| v * v).sum();
        self.window[self.index] = value;
        self.index = if self.index + 1 >= self.window.len() {
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
        let count = self.window.iter().filter(|v| **v > th).count();
        if count > 10 {
            self.counter = 100;
        }
        if self.counter > 0 {
            self.counter -= 1;
            true
        } else {
            false
        }
    }
    pub fn reset(&mut self) {
        self.window.fill(f32::NAN);
        self.counter = 0;
        self.index = 0;
    }
    pub fn new(mode: VADMode) -> VadDetector {
        VadDetector {
            index: 0,
            window: vec![f32::NAN; 50],
            counter: 0,
            mode
        }
    }
}
