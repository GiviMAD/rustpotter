pub struct MfccNormalizer {}
impl MfccNormalizer {
    pub fn normalize(frames: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let num_frames = frames.len();
        if num_frames == 0 {
            return Vec::new();
        }
        let num_mfccs = frames[0].len();
        let mut sum = Vec::with_capacity(num_mfccs);
        sum.resize(num_mfccs, 0.);
        let mut normalized_frames: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        normalized_frames.resize(num_frames, {
            let mut vector = Vec::with_capacity(num_mfccs);
            vector.resize(num_mfccs, 0.);
            vector
        });
        for (i, normalized_frames_item) in normalized_frames.iter_mut().enumerate().take(num_frames)
        {
            for (j, sum_item) in sum.iter_mut().enumerate().take(num_mfccs) {
                let value = frames[i][j];
                *sum_item += value;
                normalized_frames_item[j] = value;
            }
        }
        for normalized_frames_item in normalized_frames.iter_mut().take(num_frames) {
            for (j, sum_item) in sum.iter().enumerate().take(num_mfccs) {
                normalized_frames_item[j] -= sum_item / num_frames as f32
            }
        }
        normalized_frames
    }
}
