use rustfft::{num_complex::Complex32, FftPlanner};

use super::{MAX_I16_VAL, MIN_I16_ABS_VAL};

pub struct FeatureExtractor {
    num_coefficients: usize,
    pre_emphasis_coefficient: f32,
    samples_per_frame: usize,
    block_size: usize,
    magnitude_spectrum_size: usize,
    filter_bank: Vec<Vec<f32>>,
    hamming_window: Vec<f32>,
    // state
    samples: Vec<f32>,
}

impl FeatureExtractor {
    pub fn new(
        sample_rate: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: usize,
        pre_emphasis_coefficient: f32,
    ) -> FeatureExtractor {
        let min_frequency = 0;
        let max_frequency = sample_rate / 2;
        let magnitude_spectrum_size = samples_per_frame / 2;
        FeatureExtractor {
            samples: vec![],
            block_size: samples_per_shift * sample_rate / 8000,
            samples_per_frame,
            pre_emphasis_coefficient,
            num_coefficients,
            magnitude_spectrum_size,
            filter_bank: Self::new_mel_filter_bank(
                sample_rate,
                magnitude_spectrum_size,
                num_coefficients,
                min_frequency,
                max_frequency,
            ),
            hamming_window: Self::new_hamming_window(samples_per_frame),
        }
    }
    pub fn compute_features(&mut self, audio_samples: &[f32]) -> Vec<Vec<f32>> {
        // Remove extra bytes
        let mut samples = audio_samples.to_vec();
        let int_block_size = self.block_size / 2;
        let new_len = audio_samples.len() - (audio_samples.len() % int_block_size);
        if audio_samples.len() != new_len {
            samples.truncate(new_len);
        }
        samples
            .chunks_exact(int_block_size)
            .map(|audio_part| self.process_audio_part(audio_part))
            .filter(Option::is_some)
            .map(Option::unwrap)
            .collect()
    }
    pub fn reset(&mut self) {
        self.samples.clear();
    }
    fn process_audio_part(&mut self, audio_buffer: &[f32]) -> Option<Vec<f32>> {
        let mut new_samples = self.pre_emphasis(&audio_buffer);
        if self.samples.len() >= self.samples_per_frame {
            self.samples.drain(0..new_samples.len());
            self.samples.append(&mut new_samples);
            let features = self.extract_features(&self.samples[0..self.samples_per_frame]);
            Some(features)
        } else {
            self.samples.append(&mut new_samples);
            None
        }
    }
    fn extract_features(&self, samples: &[f32]) -> Vec<f32> {
        let magnitude_spectrum = self.calculate_magnitude_spectrum(samples);
        let mut features = self.calculate_mel_frequency_cepstral_coefficients(&magnitude_spectrum);
        features.drain(0..1);
        features
    }
    fn pre_emphasis(&self, audio_buffer: &[f32]) -> Vec<f32> {
        let mut samples = vec![];
        for i in 0..audio_buffer.len() {
            let current = audio_buffer[i];
            let previous = if i != 0 { audio_buffer[i - 1] } else { 0. };
            let sample_with_pre_emphasis =
                current as f32 - self.pre_emphasis_coefficient * previous as f32;
            // convert to float sample
            let float_sample = if sample_with_pre_emphasis < 0. {
                sample_with_pre_emphasis / MIN_I16_ABS_VAL
            } else {
                sample_with_pre_emphasis / MAX_I16_VAL
            };
            samples.push(float_sample.clamp(-1., 1.));
        }
        samples
    }

    //==================================================================
    // Feature extraction utils
    fn calculate_magnitude_spectrum(&self, audio_frame: &[f32]) -> Vec<f32> {
        let mut buffer = vec![];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.samples_per_frame);
        for _i in 0..self.samples_per_frame {
            buffer.push(Complex32 {
                re: audio_frame[_i] * self.hamming_window[_i],
                im: 0.,
            });
        }
        fft.process(&mut buffer);
        let mut magnitude_spectrum: Vec<f32> = Vec::with_capacity(buffer.len());
        for _i in 0..self.magnitude_spectrum_size {
            magnitude_spectrum
                .push(((buffer[_i].re * buffer[_i].re) + (buffer[_i].im * buffer[_i].im)).sqrt());
        }
        magnitude_spectrum
    }
    fn new_hamming_window(samples_per_frame: usize) -> Vec<f32> {
        let ns_minus_1 = samples_per_frame - 1;
        (0..samples_per_frame)
            .map(|s| {
                0.54 - (0.46 * (2. * std::f32::consts::PI * (s as f32 / ns_minus_1 as f32)).cos())
            })
            .collect()
    }
    fn calculate_mel_frequency_cepstral_coefficients(
        &self,
        magnitude_spectrum: &[f32],
    ) -> Vec<f32> {
        let mel_spectrum = self.calculate_mel_frequency_spectrum(magnitude_spectrum);
        let mut mfccs = Vec::new();
        for i in 0..mel_spectrum.len() {
            let value = (mel_spectrum[i] + f32::MIN_POSITIVE).ln();
            mfccs.push(value);
        }
        mfccs = self.discrete_cosine_transform(mfccs);
        mfccs
    }
    fn frequency_to_mel(frequency: usize) -> f32 {
        return 1127. * (1. + (frequency as f32 / 700.0)).ln();
    }
    fn calculate_mel_frequency_spectrum(&self, magnitude_spectrum: &[f32]) -> Vec<f32> {
        let mut mel_spectrum: Vec<f32> = Vec::new();
        for i in 0..self.num_coefficients {
            let mut coeff = 0.;
            for j in 0..magnitude_spectrum.len() {
                let filter_bank_val = self.filter_bank[i][j];
                coeff += (magnitude_spectrum[j] * magnitude_spectrum[j]) * filter_bank_val;
            }
            mel_spectrum.push(coeff);
        }
        mel_spectrum
    }
    fn discrete_cosine_transform(&self, mut input_signal: Vec<f32>) -> Vec<f32> {
        // the input signal must have the number of elements specified in the numElements variable
        let num_elements = input_signal.len();
        let mut dct_signal: Vec<f32> = Vec::new();
        for i in 0..num_elements {
            dct_signal.push(input_signal[i]);
        }
        let pi_over_n = std::f32::consts::PI / num_elements as f32;

        for k in 0..num_elements {
            let mut sum = 0.;
            for n in 0..num_elements {
                let tmp = pi_over_n * (n as f32 + 0.5) * k as f32;
                sum += dct_signal[n] * tmp.cos();
            }
            input_signal[k] = 2. * sum;
        }
        input_signal
    }
    fn new_mel_filter_bank(
        sample_rate: usize,
        magnitude_spectrum_size: usize,
        num_coefficients: usize,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Vec<Vec<f32>> {
        let max_mel = Self::frequency_to_mel(max_frequency).floor();
        let min_mel = Self::frequency_to_mel(min_frequency).floor();

        let mut filter_bank = vec![vec![0.; magnitude_spectrum_size]; num_coefficients];
        let mut centre_indices: Vec<usize> = Vec::new();

        for i in 0..num_coefficients + 2 {
            let f = i as f32 * (max_mel - min_mel) as f32 / (num_coefficients + 1) as f32 + min_mel;
            let mut tmp = (1 as f32 + 1000.0 / 700.0).ln() / 1000.0;
            tmp = ((f as f32 * tmp).exp() - 1.) / (sample_rate as f32 / 2 as f32);
            tmp = 0.5 + 700. * magnitude_spectrum_size as f32 * tmp;
            let centre_index = tmp.floor() as usize;
            centre_indices.push(centre_index);
        }
        for i in 0..num_coefficients {
            let filter_begin_index = centre_indices[i];
            let filter_center_index = centre_indices[i + 1];
            let filter_end_index = centre_indices[i + 2];

            let triangle_range_up = filter_center_index - filter_begin_index;
            let triangle_range_down = filter_end_index - filter_center_index;

            // upward slope
            for k in filter_begin_index..filter_center_index {
                filter_bank[i][k] = (k - filter_begin_index) as f32 / triangle_range_up as f32;
            }
            // downwards slope
            for k in filter_center_index..filter_end_index {
                filter_bank[i][k] = (filter_end_index - k) as f32 / triangle_range_down as f32;
            }
        }
        filter_bank
    }
}
