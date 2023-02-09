use rustfft::{num_complex::Complex32, FftPlanner};
use simple_matrix::Matrix;
use std::iter;
pub struct FeatureExtractor {
    num_coefficients: usize,
    pre_emphasis_coefficient: f32,
    sample_rate: usize,
    samples_per_frame: usize,
    block_size: usize,
    samples: Vec<f32>,
    filter_bank: Option<Matrix<f32>>,
    hamming_window: Option<Vec<f32>>,
    min_frequency: usize,
    max_frequency: usize,
}
impl FeatureExtractor {
    pub fn new(
        sample_rate: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: usize,
        pre_emphasis_coefficient: f32,
    ) -> Self {
        let mut extractor = FeatureExtractor {
            samples: vec![],
            sample_rate,
            block_size: samples_per_shift * sample_rate / 8000,
            samples_per_frame, // TODO: add options
            pre_emphasis_coefficient,
            num_coefficients,
            min_frequency: 0,
            max_frequency: sample_rate / 2,
            filter_bank: None,
            hamming_window: None,
        };
        extractor.calculate_mel_filter_bank();
        extractor.create_hamming_window();
        extractor
    }
    fn get_magnitude_spectrum_size(&self) -> usize {
        self.samples_per_frame / 2
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
            let transformed = convert_int16_to_float32(
                current as f32 - self.pre_emphasis_coefficient * previous as f32,
            );
            samples.push(transformed)
        }

        samples
    }

    //==================================================================
    // ported from Gist C++ audio library
    fn calculate_magnitude_spectrum(&self, audio_frame: &[f32]) -> Vec<f32> {
        if self.hamming_window.is_none() {
            panic!("[Extractor] hamming window not initialized");
        }
        let mut buffer = vec![];
        let window = self.hamming_window.as_ref().unwrap();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.samples_per_frame);
        for _i in 0..self.samples_per_frame {
            buffer.push(Complex32 {
                re: audio_frame[_i] * window[_i],
                im: 0.,
            });
        }
        fft.process(&mut buffer);
        let mut magnitude_spectrum: Vec<f32> = Vec::with_capacity(buffer.len());
        for _i in 0..self.get_magnitude_spectrum_size() {
            magnitude_spectrum
                .push(((buffer[_i].re * buffer[_i].re) + (buffer[_i].im * buffer[_i].im)).sqrt());
        }
        magnitude_spectrum
    }
    fn create_hamming_window(&mut self) {
        let mut window = vec![];
        let num_samples_minus_1 = self.samples_per_frame - 1; // the number of samples minus 1
        for i in 0..self.samples_per_frame {
            window.push(
                0.54 - (0.46
                    * (2. * std::f32::consts::PI * (i as f32 / num_samples_minus_1 as f32)).cos()),
            );
        }
        self.hamming_window = Some(window);
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
    fn frequency_to_mel(&self, frequency: usize) -> f32 {
        return 1127. * (1. + (frequency as f32 / 700.0)).ln();
    }
    fn calculate_mel_frequency_spectrum(&self, magnitude_spectrum: &[f32]) -> Vec<f32> {
        if self.filter_bank.is_none() {
            panic!("[Extractor] filter bank not initialized");
        }
        let filter_bank = self.filter_bank.as_ref().unwrap();
        let mut mel_spectrum: Vec<f32> = Vec::new();
        for i in 0..self.num_coefficients {
            let mut coeff = 0.;
            for j in 0..magnitude_spectrum.len() {
                let filter_bank_val = filter_bank.get(i, j).unwrap();
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
    fn calculate_mel_filter_bank(&mut self) {
        let max_mel = self.frequency_to_mel(self.max_frequency).floor();
        let min_mel = self.frequency_to_mel(self.min_frequency).floor();

        let mut filter_bank: Matrix<f32> = Matrix::from_iter(
            self.num_coefficients,
            self.get_magnitude_spectrum_size(),
            iter::repeat(0.),
        );

        let mut centre_indices: Vec<usize> = Vec::new();

        for i in 0..self.num_coefficients + 2 {
            let f = i as f32 * (max_mel - min_mel) as f32 / (self.num_coefficients + 1) as f32
                + min_mel;
            let mut tmp = (1 as f32 + 1000.0 / 700.0).ln() / 1000.0;
            tmp = ((f as f32 * tmp).exp() - 1.) / (self.sample_rate as f32 / 2 as f32);
            tmp = 0.5 + 700. * self.get_magnitude_spectrum_size() as f32 * tmp;
            let centre_index = tmp.floor() as usize;
            centre_indices.push(centre_index);
        }
        for i in 0..self.num_coefficients {
            let filter_begin_index = centre_indices[i];
            let filter_center_index = centre_indices[i + 1];
            let filter_end_index = centre_indices[i + 2];

            let triangle_range_up = filter_center_index - filter_begin_index;
            let triangle_range_down = filter_end_index - filter_center_index;

            // upward slope
            for k in filter_begin_index..filter_center_index {
                filter_bank.set(
                    i,
                    k,
                    (k - filter_begin_index) as f32 / triangle_range_up as f32,
                );
            }
            // downwards slope
            for k in filter_center_index..filter_end_index {
                filter_bank.set(
                    i,
                    k,
                    (filter_end_index - k) as f32 / triangle_range_down as f32,
                );
            }
        }
        self.filter_bank = Option::Some(filter_bank)
    }
}

pub fn convert_int16_to_float32(n: f32) -> f32 {
    let v = if n < 0. { n / 32768. } else { n / 32767. }; // convert in range [-32768, 32767]
    return v.clamp(-1., 1.);
}
