use std::f32::consts::PI;

use rustfft::{num_complex::Complex32, FftPlanner};

pub struct MfccExtractor {
    num_coefficients: usize,
    pre_emphasis_coefficient: f32,
    samples_per_frame: usize,
    samples_per_shift: usize,
    magnitude_spectrum_size: usize,
    sample_rate: usize,
    // state
    filter_bank: Vec<Vec<f32>>,
    hamming_window: Vec<f32>,
    samples: Vec<f32>,
}

impl MfccExtractor {
    pub fn new(
        sample_rate: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: u16,
        pre_emphasis_coefficient: f32,
    ) -> MfccExtractor {
        let min_frequency = 0;
        let max_frequency = sample_rate / 2;
        let magnitude_spectrum_size = samples_per_frame / 2;
        MfccExtractor {
            samples: vec![],
            samples_per_shift,
            samples_per_frame,
            pre_emphasis_coefficient,
            num_coefficients: num_coefficients as usize,
            magnitude_spectrum_size,
            filter_bank: Self::new_mel_filter_bank(
                sample_rate,
                magnitude_spectrum_size,
                num_coefficients as usize,
                min_frequency,
                max_frequency,
            ),
            hamming_window: Self::new_hamming_window(samples_per_frame),
            sample_rate,
        }
    }
    pub fn set_out_size(&mut self, out_size: u16) {
        self.num_coefficients = (out_size + 1) as usize; // first coefficient is dropped
        let min_frequency = 0;
        let max_frequency = self.sample_rate / 2;
        self.filter_bank = Self::new_mel_filter_bank(
            self.sample_rate,
            self.magnitude_spectrum_size,
            self.num_coefficients,
            min_frequency,
            max_frequency,
        );
        self.reset();
    }
    pub fn compute(&mut self, audio_samples: &[f32]) -> Vec<Vec<f32>> {
        audio_samples
            .chunks_exact(self.samples_per_shift)
            .filter_map(|audio_part| self.process_audio_part(audio_part))
            .collect()
    }
    pub fn reset(&mut self) {
        self.samples.clear();
    }
    fn process_audio_part(&mut self, audio_buffer: &[f32]) -> Option<Vec<f32>> {
        let mut new_samples = self.pre_emphasis(audio_buffer);
        if self.samples.len() >= self.samples_per_frame {
            self.samples.drain(0..new_samples.len());
            self.samples.append(&mut new_samples);
            Some(self.extract_mfccs(&self.samples[0..self.samples_per_frame]))
        } else {
            self.samples.append(&mut new_samples);
            None
        }
    }
    fn extract_mfccs(&self, samples: &[f32]) -> Vec<f32> {
        let magnitude_spectrum = self.calculate_magnitude_spectrum(samples);
        let mut mfcc_frame =
            self.calculate_mel_frequency_cepstral_coefficients(&magnitude_spectrum);
        mfcc_frame.drain(0..1);
        mfcc_frame
    }
    fn pre_emphasis(&self, audio_buffer: &[f32]) -> Vec<f32> {
        let mut tmp_sample = 0.;
        audio_buffer
            .iter()
            .map(|current| {
                let previous = tmp_sample;
                tmp_sample = *current;
                tmp_sample - self.pre_emphasis_coefficient * previous
            })
            .collect()
    }

    //==================================================================
    // Feature extraction utils
    fn calculate_magnitude_spectrum(&self, audio_frame: &[f32]) -> Vec<f32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.samples_per_frame);
        let mut buffer = (0..self.samples_per_frame)
            .map(|i| Complex32 {
                re: audio_frame[i] * self.hamming_window[i],
                im: 0.,
            })
            .collect::<Vec<_>>();
        fft.process(&mut buffer);
        (0..self.magnitude_spectrum_size)
            .map(|i| ((buffer[i].re * buffer[i].re) + (buffer[i].im * buffer[i].im)).sqrt())
            .collect()
    }
    fn new_hamming_window(samples_per_frame: usize) -> Vec<f32> {
        let ns_minus_1 = samples_per_frame - 1;
        (0..samples_per_frame)
            .map(|s| 0.54 - (0.46 * (2. * PI * (s as f32 / ns_minus_1 as f32)).cos()))
            .collect()
    }
    fn calculate_mel_frequency_cepstral_coefficients(
        &self,
        magnitude_spectrum: &[f32],
    ) -> Vec<f32> {
        let mfccs: Vec<f32> = self
            .calculate_mel_frequency_cepstrum(magnitude_spectrum)
            .iter()
            .map(|ms| (ms + f32::MIN_POSITIVE).ln())
            .collect();
        self.discrete_cosine_transform(mfccs)
    }
    fn frequency_to_mel(frequency: usize) -> f32 {
        1127. * (1. + (frequency as f32 / 700.0)).ln()
    }
    fn calculate_mel_frequency_cepstrum(&self, magnitude_spectrum: &[f32]) -> Vec<f32> {
        (0..self.num_coefficients)
            .map(|i| {
                magnitude_spectrum
                    .iter()
                    .enumerate()
                    .map(|(j, ms)| ms * ms * self.filter_bank[i][j])
                    .sum()
            })
            .collect()
    }
    fn discrete_cosine_transform(&self, mut input_signal: Vec<f32>) -> Vec<f32> {
        let dct_signal: Vec<f32> = input_signal.to_vec();
        let pi_over_n = PI / input_signal.len() as f32;
        input_signal
            .iter_mut()
            .enumerate()
            .for_each(|(k, input_sample)| {
                *input_sample = 2.
                    * dct_signal
                        .iter()
                        .enumerate()
                        .map(|(n, dct_sample)| {
                            dct_sample * (pi_over_n * (n as f32 + 0.5) * k as f32).cos()
                        })
                        .sum::<f32>();
            });
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
        let centre_indices: Vec<usize> = (0..num_coefficients + 2)
            .map(|i| {
                let f = i as f32 * (max_mel - min_mel) / (num_coefficients + 1) as f32 + min_mel;
                let mut tmp = (1_f32 + 1000.0 / 700.0).ln() / 1000.0;
                tmp = ((f * tmp).exp() - 1.) / (sample_rate as f32 / 2.);
                (0.5 + 700. * magnitude_spectrum_size as f32 * tmp).floor() as usize
            })
            .collect();
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
