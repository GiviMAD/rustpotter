use std::{cmp::Ordering, collections::HashMap};

use crate::{
    internal::{
        BandPassFilter, FeatureComparator, FeatureExtractor, FeatureNormalizer,
        GainNormalizerFilter, WAVEncoder,
    },
    RustpotterConfig, Wakeword, DETECTOR_INTERNAL_BIT_DEPTH, DETECTOR_INTERNAL_SAMPLE_RATE,
    FEATURE_EXTRACTOR_FRAME_LENGTH_MS, FEATURE_EXTRACTOR_FRAME_SHIFT_MS,
    FEATURE_EXTRACTOR_NUM_COEFFICIENT, FEATURE_EXTRACTOR_PRE_EMPHASIS,
};

pub struct Rustpotter {
    // Config
    /// Required score against the averaged features vector. Only for discard frames.
    avg_threshold: f32,
    /// Required score while comparing the wakeword against the audio_features_window.
    threshold: f32,
    /// Required number of partial scores (scores over threshold) to consider the detection real.
    min_scores: usize,
    // Utils
    /// Utility to encode or re-encode the input wav data.
    wav_encoder: WAVEncoder,
    /// Utility to extract a collection of features for each input audio frame.
    feature_extractor: FeatureExtractor,
    /// Utility to measure the similarity between two feature frame vectors.
    feature_comparator: FeatureComparator,
    /// Optional band-pass filter implementation.
    band_pass_filter: Option<BandPassFilter>,
    /// Optional gain filter implementation.
    gain_normalizer_filter: Option<GainNormalizerFilter>,
    // State
    /// The collection of wakewords to run detection against.
    wakewords: Vec<Wakeword>,
    /// Indicates that the audio_features_window has enough features to run the detection.
    /// This means its length is greater or equal to the min_features_frames value.
    buffering: bool,
    /// A window of feature frames extract from the input audio.
    /// It grows until match the max_features_frames value.
    audio_features_window: Vec<Vec<f32>>,
    /// Min number of feature frames on the wakewords.
    min_features_frames: usize,
    /// Max number of feature frames on the wakewords.
    max_features_frames: usize,
    /// Stores the partial detection with greater score while waiting for the detection countdown to be zero.
    partial_detection: Option<RustpotterDetection>,
    /// Countdown until detection fires.
    /// Whenever a better partial detection is found, it is set to the double of the max number of feature frames in the wakeword samples.
    /// When it gets to zero and a partial detection exists, it'll be considered the final detection.
    detection_countdown: usize,
}

impl Rustpotter {
    /// Returns a configured Rustpotter instance.
    pub fn new(config: RustpotterConfig) -> Result<Rustpotter, &'static str> {
        let reencoder = WAVEncoder::new(
            &config.fmt,
            FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
            DETECTOR_INTERNAL_BIT_DEPTH,
        )?;
        let samples_per_frame = reencoder.get_output_frame_length();
        let samples_per_shift = (samples_per_frame as f32
            / (FEATURE_EXTRACTOR_FRAME_LENGTH_MS as f32 / FEATURE_EXTRACTOR_FRAME_SHIFT_MS as f32)
                as f32) as usize;
        let feature_extractor = FeatureExtractor::new(
            config.fmt.sample_rate,
            samples_per_frame,
            samples_per_shift,
            FEATURE_EXTRACTOR_NUM_COEFFICIENT,
            FEATURE_EXTRACTOR_PRE_EMPHASIS,
        );
        let feature_comparator = FeatureComparator::new(
            config.detector.comparator_band_size,
            config.detector.comparator_reference,
        );
        let band_pass_filter = if config.filters.band_pass {
            Some(BandPassFilter::new(
                config.fmt.sample_rate as f32,
                config.filters.low_cutoff,
                config.filters.high_cutoff,
            ))
        } else {
            None
        };
        let gain_normalizer_filter = if config.filters.gain_normalizer {
            Some(GainNormalizerFilter::new())
        } else {
            None
        };
        Ok(Rustpotter {
            avg_threshold: config.detector.avg_threshold,
            threshold: config.detector.threshold,
            min_scores: config.detector.min_scores,
            wav_encoder: reencoder,
            feature_extractor,
            feature_comparator,
            band_pass_filter,
            gain_normalizer_filter,
            audio_features_window: Vec::new(),
            buffering: true,
            max_features_frames: 0,
            min_features_frames: 0,
            wakewords: Vec::new(),
            partial_detection: None,
            detection_countdown: 0,
        })
    }
    /// Add wakeword
    pub fn add_wakeword(&mut self, wakeword: Wakeword) {
        self.wakewords.push(wakeword);
        // update detection window requirements and gain normalizer target rms level.
        let mut min_feature_frames = usize::MAX;
        let mut max_feature_frames = usize::MIN;
        let mut rms_level = f32::NAN;
        for wakeword in self.wakewords.iter() {
            if !wakeword.samples_features.is_empty() {
                min_feature_frames = wakeword
                    .samples_features
                    .iter()
                    .map(|(_, feature_frames)| feature_frames.len())
                    .min()
                    .unwrap_or(0)
                    .min(min_feature_frames);
                max_feature_frames = wakeword
                    .samples_features
                    .iter()
                    .map(|(_, feature_frames)| feature_frames.len())
                    .max()
                    .unwrap_or(0)
                    .max(max_feature_frames);
                rms_level = wakeword.rms_level.max(rms_level);
            }
        }
        self.min_features_frames = min_feature_frames;
        self.max_features_frames = max_feature_frames;
        if self.gain_normalizer_filter.is_some() {
            self.gain_normalizer_filter
                .as_mut()
                .unwrap()
                .target_rms_level = rms_level;
        }
        self.buffering = self.audio_features_window.len() < self.min_features_frames;
    }
    /// Add wakeword from model bytes
    pub fn add_wakeword_from_buffer(&mut self, buffer: &[u8]) -> Result<(), String> {
        Ok(self.add_wakeword(Wakeword::load_from_buffer(buffer)?))
    }
    /// Add wakeword from model path
    pub fn add_wakeword_from_file(&mut self, path: &str) -> Result<(), String> {
        Ok(self.add_wakeword(Wakeword::load_from_file(path)?))
    }
    /// Returns the number of audio samples needed by the detector.
    pub fn get_samples_per_frame(&self) -> usize {
        self.wav_encoder.get_input_frame_length()
    }
    /// Returns the number of audio bytes needed by the detector.
    pub fn get_bytes_per_frame(&self) -> usize {
        self.wav_encoder.get_input_byte_length()
    }
    /// Process bytes buffer.
    ///
    /// Requires that the buffer length matches the return
    /// of the get_bytes_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Assumes buffer endianness matches the configured for the detector.
    ///
    pub fn process_byte_buffer(&mut self, audio_bytes: Vec<u8>) -> Option<RustpotterDetection> {
        let audio_buffer_encoded = self.wav_encoder.encode(audio_bytes);
        self.process_internal(audio_buffer_encoded)
    }
    /// Process i32 audio chunks.
    ///
    /// Requires that the audio chunk length matches the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Assumes that detector bits_per_sample is one of: 8, 16.
    ///
    /// Assumes that detector sample_format is 'int'.
    ///
    /// It's an alias for the process_i32 method.
    pub fn process_short_int_buffer(
        &mut self,
        audio_samples: Vec<i16>,
    ) -> Option<RustpotterDetection> {
        let audio_buffer_encoded = self.wav_encoder.reencode(
            audio_samples
                .into_iter()
                .map(|sample| sample as i32)
                .collect(),
        );
        self.process_internal(audio_buffer_encoded)
    }
    /// Process i32 audio chunks.
    ///
    /// Requires that the audio chunk length matches the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Assumes that detector bits_per_sample is one of: 8, 16, 24, 32.
    ///
    /// Assumes that detector sample_format is 'int'.
    ///
    /// It's an alias for the process_i32 method.
    pub fn process_int_buffer(&mut self, audio_samples: Vec<i32>) -> Option<RustpotterDetection> {
        let audio_buffer_encoded = self.wav_encoder.reencode(audio_samples);
        self.process_internal(audio_buffer_encoded)
    }
    /// Process f32 audio chunks.
    ///
    /// Requires that the audio chunk length matches the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Requires that detector bits_per_sample is 32.
    ///
    /// Requires that detector sample_format is 'float'.
    pub fn process_float_buffer(&mut self, audio_samples: Vec<f32>) -> Option<RustpotterDetection> {
        let audio_buffer_encoded = self.wav_encoder.reencode_float(audio_samples);
        self.process_internal(audio_buffer_encoded)
    }
    fn process_internal(&mut self, audio_buffer: Vec<f32>) -> Option<RustpotterDetection> {
        let mut processed_audio = audio_buffer;
        if self.gain_normalizer_filter.is_some() {
            self.gain_normalizer_filter
                .as_mut()
                .unwrap()
                .filter(&mut processed_audio);
        }
        if self.band_pass_filter.is_some() {
            self.band_pass_filter
                .as_mut()
                .unwrap()
                .filter(&mut processed_audio);
        }
        self.feature_extractor
            .compute_features(&processed_audio)
            .into_iter()
            .find_map(|features| self.process_new_features(features))
    }
    fn process_new_features(&mut self, features: Vec<f32>) -> Option<RustpotterDetection> {
        let mut result: Option<RustpotterDetection> = None;
        self.audio_features_window.push(features);
        if self.audio_features_window.len() >= self.min_features_frames {
            if self.buffering {
                self.buffering = false;
            }
            result = self.run_detection();
        }
        if self.audio_features_window.len() >= self.max_features_frames {
            self.audio_features_window.drain(0..1);
        }
        result
    }
    fn run_detection(&mut self) -> Option<RustpotterDetection> {
        // let result_option = self.get_best_wakeword(self.audio_features.to_vec());
        // match result_option {}
        let mut wakeword_detections = self
            .wakewords
            .iter()
            .filter_map(|wakeword| {
                if !wakeword.enabled {
                    return None;
                }
                let avg_threshold = wakeword.avg_threshold.unwrap_or(self.avg_threshold);
                let mut avg_features_score = 0.;
                if wakeword.avg_features.is_some() && avg_threshold != 0. {
                    // discard detections against the wakeword averaged features
                    let wakeword_samples_avg_features = wakeword.avg_features.as_ref().unwrap();
                    let frame_features_normalized = self.cut_and_normalize_frame(
                        self.audio_features_window.to_vec(),
                        wakeword_samples_avg_features.len(),
                    );
                    avg_features_score =
                        self.score_frame(&frame_features_normalized, wakeword_samples_avg_features);
                    if avg_features_score < avg_threshold {
                        return None;
                    }
                }
                let threshold = wakeword.threshold.unwrap_or(self.threshold);
                let wakeword_scores = wakeword.samples_features.iter().fold(
                    HashMap::new(),
                    |mut acc: HashMap<String, f32>, (name, wakeword_sample_features)| {
                        let frame_features_normalized = self.cut_and_normalize_frame(
                            self.audio_features_window.to_vec(),
                            wakeword_sample_features.len(),
                        );
                        acc.insert(
                            name.to_string(),
                            self.score_frame(&frame_features_normalized, wakeword_sample_features),
                        );
                        acc
                    },
                );
                let mut scores = wakeword_scores
                    .iter()
                    .map(|(_, score)| *score)
                    .collect::<Vec<f32>>();
                scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                let max_score = scores[0];
                if max_score > threshold {
                    Some(RustpotterDetection {
                        name: wakeword.name.to_string(),
                        avg_score: avg_features_score,
                        score: max_score,
                        scores: wakeword_scores,
                        counter: self.partial_detection.as_ref().map_or(1, |d| d.counter + 1),
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<RustpotterDetection>>();
        wakeword_detections
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        let wakeword_detection = wakeword_detections.into_iter().next();
        if self.detection_countdown != 0 {
            self.detection_countdown -= 1;
        }
        if self.partial_detection.is_some() && self.detection_countdown == 0 {
            let wakeword_detection = self.partial_detection.take().unwrap();
            if wakeword_detection.counter > self.min_scores {
                self.buffering = true;
                self.audio_features_window.clear();
                Some(wakeword_detection)
            } else {
                None
            }
        } else {
            if wakeword_detection.is_some() {
                if self.partial_detection.is_none()
                    || self.partial_detection.as_ref().unwrap().score
                        < wakeword_detection.as_ref().unwrap().score
                {
                    self.partial_detection = wakeword_detection;
                } else {
                    self.partial_detection.as_mut().unwrap().counter =
                        wakeword_detection.as_ref().unwrap().counter;
                }
                self.detection_countdown = self.max_features_frames * 2;
            }
            None
        }
    }
    fn cut_and_normalize_frame(
        &self,
        mut features: Vec<Vec<f32>>,
        max_len: usize,
    ) -> Vec<Vec<f32>> {
        if features.len() > max_len {
            features.drain(max_len..features.len());
        }
        FeatureNormalizer::normalize(features)
    }
    fn score_frame(&self, frame_features: &Vec<Vec<f32>>, template: &Vec<Vec<f32>>) -> f32 {
        let score = self.feature_comparator.compare(
            &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            &frame_features
                .iter()
                .map(|item| &item[..])
                .collect::<Vec<_>>(),
        );
        score
    }
}
pub struct RustpotterDetection {
    /// Detected wakeword name.
    pub name: String,
    /// Detection score against the averaged template.
    pub avg_score: f32,
    /// Detection score
    pub score: f32,
    /// Detection scores against each template
    pub scores: HashMap<String, f32>,
    /// Partial detections counter
    pub counter: usize,
}
