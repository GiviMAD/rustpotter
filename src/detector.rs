use std::{cmp::Ordering, collections::HashMap};

use crate::{
    internal::{
        BandPassFilter, FeatureComparator, FeatureExtractor, FeatureNormalizer,
        GainNormalizerFilter, WAVEncoder,
    },
    RustpotterConfig, ScoreMode, Wakeword, DETECTOR_INTERNAL_SAMPLE_RATE,
    FEATURE_EXTRACTOR_FRAME_LENGTH_MS, FEATURE_EXTRACTOR_FRAME_SHIFT_MS,
    FEATURE_EXTRACTOR_NUM_COEFFICIENT, FEATURE_EXTRACTOR_PRE_EMPHASIS,
};
/// Rustpotter is an open source wakeword spotter forged in rust
/// ```
/// use rustpotter::{Rustpotter, RustpotterConfig, Wakeword};
/// // assuming the audio input format match the detector defaults
/// let mut detector_config = RustpotterConfig::default();
/// // Configure the detector
/// // ...
/// let mut detector = Rustpotter::new(&detector_config).unwrap();
/// // load and enable a wakeword
/// detector.add_wakeword_from_file("./tests/resources/oye_casa_g.rpw").unwrap();
/// // You need a buffer of size detector.get_samples_per_frame() when using samples or detector.get_bytes_per_frame() when using bytes.  
/// let mut frame_buffer: Vec<i16> = vec![0; detector.get_samples_per_frame()];
/// // while true { Iterate forever
///     // fill the buffer with the required samples/bytes...
///     let detection_option = detector.process_short_buffer(&frame_buffer);
///     if detection_option.is_some() {
///         let detection = detection_option.unwrap();
///         // println!("{:?}", detection);
///     }
/// // }
/// ```
pub struct Rustpotter {
    // Config
    /// Required score against the averaged features vector. Only for discard frames.
    avg_threshold: f32,
    /// Required score while comparing the wakeword against the audio_features_window.
    threshold: f32,
    /// Required number of partial scores (scores over threshold) to consider the detection real.
    min_scores: usize,
    /// How to calculate the final score.
    score_mode: ScoreMode,
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
    /// This means its length is greater or equal to the max_features_frames value.
    buffering: bool,
    /// A window of feature frames extract from the input audio.
    /// It grows until match the max_features_frames value.
    audio_features_window: Vec<Vec<f32>>,
    /// Max number of feature frames on the wakewords.
    max_features_frames: usize,
    /// Stores the partial detection with greater score while waiting for the detection countdown to be zero.
    partial_detection: Option<RustpotterDetection>,
    /// Countdown until detection fires.
    /// Whenever a better partial detection is found, it is set to the double of the max number of feature frames in the wakeword samples.
    /// When it gets to zero and a partial detection exists, it'll be considered the final detection.
    detection_countdown: usize,
    /// Current frame rms level
    rms_level: f32,
    /// Gain normalization applied to current frame, 1 if none
    gain: f32,
}

impl Rustpotter {
    /// Returns a configured Rustpotter instance.
    pub fn new(config: &RustpotterConfig) -> Result<Rustpotter, &'static str> {
        let reencoder = WAVEncoder::new(
            &config.fmt,
            FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )?;
        let samples_per_frame = reencoder.get_output_frame_length();
        let samples_per_shift = (samples_per_frame as f32
            / (FEATURE_EXTRACTOR_FRAME_LENGTH_MS as f32 / FEATURE_EXTRACTOR_FRAME_SHIFT_MS as f32)
                as f32) as usize;
        let feature_extractor = FeatureExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
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
            score_mode: config.detector.score_mode,
            wav_encoder: reencoder,
            feature_extractor,
            feature_comparator,
            band_pass_filter,
            gain_normalizer_filter,
            audio_features_window: Vec::new(),
            buffering: true,
            max_features_frames: 0,
            wakewords: Vec::new(),
            partial_detection: None,
            detection_countdown: 0,
            rms_level: 0.,
            gain: 0.,
        })
    }
    /// Add wakeword
    pub fn add_wakeword(&mut self, wakeword: Wakeword) {
        self.wakewords.push(wakeword);
        // update detection window requirements and gain normalizer target rms level
        let mut max_feature_frames = usize::MIN;
        let mut target_rms_level = f32::NAN;
        for wakeword in self.wakewords.iter() {
            if !wakeword.samples_features.is_empty() {
                max_feature_frames = wakeword
                    .samples_features
                    .iter()
                    .map(|(_, feature_frames)| feature_frames.len())
                    .max()
                    .unwrap_or(0)
                    .max(max_feature_frames);
                target_rms_level = wakeword.rms_level.max(target_rms_level);
            }
        }
        self.max_features_frames = max_feature_frames;
        if let Some(gain_normalizer_filter) = self.gain_normalizer_filter.as_mut() {
            gain_normalizer_filter.target_rms_level = target_rms_level;
        }
        self.buffering = self.audio_features_window.len() < self.max_features_frames;
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
    /// Returns a reference to the current partial detection if any.
    pub fn get_partial_detection(&self) -> Option<&RustpotterDetection> {
        self.partial_detection.as_ref()
    }
    /// Returns the rms level of the last frame (before gain normalization)
    pub fn get_rms_level(&self) -> f32 {
        self.rms_level
    }
    /// Returns the gain normalization applied to the latest frame
    pub fn get_gain_normalization(&self) -> f32 {
        self.gain
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
    pub fn process_byte_buffer(&mut self, audio_bytes: &[u8]) -> Option<RustpotterDetection> {
        let mut encoded_samples = self.wav_encoder.encode(audio_bytes);
        self.process_internal(&mut encoded_samples)
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
    pub fn process_short_buffer(&mut self, audio_samples: &[i16]) -> Option<RustpotterDetection> {
        let mut encoded_samples = self.wav_encoder.reencode_int(
            &audio_samples
                .into_iter()
                .map(|sample| *sample as i32)
                .collect::<Vec<i32>>(),
        );
        self.process_internal(&mut encoded_samples)
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
    pub fn process_int_buffer(&mut self, audio_samples: &[i32]) -> Option<RustpotterDetection> {
        let mut encoded_samples = self.wav_encoder.reencode_int(audio_samples);
        self.process_internal(&mut encoded_samples)
    }
    /// Process f32 audio chunks.
    ///
    /// Requires that the audio chunk length matches the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Requires that detector bits_per_sample is 32 to work.
    ///
    /// Requires that detector sample_format is 'float' to work.
    pub fn process_float_buffer(&mut self, audio_samples: &[f32]) -> Option<RustpotterDetection> {
        let mut encoded_samples = self.wav_encoder.reencode_float(audio_samples);
        self.process_internal(&mut encoded_samples)
    }
    fn process_internal(&mut self, audio_buffer: &mut [f32]) -> Option<RustpotterDetection> {
        self.rms_level = GainNormalizerFilter::get_rms_level(&audio_buffer);
        if self.gain_normalizer_filter.is_some() {
            self.gain = self
                .gain_normalizer_filter
                .as_mut()
                .unwrap()
                .filter(audio_buffer, self.rms_level);
        }
        if self.band_pass_filter.is_some() {
            self.band_pass_filter.as_mut().unwrap().filter(audio_buffer);
        }
        self.feature_extractor
            .compute_features(audio_buffer)
            .into_iter()
            .find_map(|features| self.process_new_features(features))
    }
    fn process_new_features(&mut self, features_frame: Vec<f32>) -> Option<RustpotterDetection> {
        let mut result: Option<RustpotterDetection> = None;
        self.audio_features_window.push(features_frame);
        if self.audio_features_window.len() >= self.max_features_frames {
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
        if self.detection_countdown != 0 {
            self.detection_countdown -= 1;
        }
        let mut wakeword_detections = self
            .wakewords
            .iter()
            .filter_map(|wakeword| {
                if !wakeword.enabled {
                    return None;
                }
                let avg_threshold = wakeword.avg_threshold.unwrap_or(self.avg_threshold);
                let mut avg_score = 0.;
                if wakeword.avg_features.is_some() && avg_threshold != 0. {
                    // discard detections against the wakeword averaged features
                    let wakeword_samples_avg_features = wakeword.avg_features.as_ref().unwrap();
                    let audio_window_normalized = self.cut_and_normalize_frame(
                        self.audio_features_window.to_vec(),
                        wakeword_samples_avg_features.len(),
                    );
                    avg_score =
                        self.score_frame(&audio_window_normalized, wakeword_samples_avg_features);
                    if avg_score < avg_threshold {
                        return None;
                    }
                }
                let threshold = wakeword.threshold.unwrap_or(self.threshold);
                let scores = wakeword.samples_features.iter().fold(
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
                let mut sorted_scores =
                    scores.iter().map(|(_, score)| *score).collect::<Vec<f32>>();
                sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                let score = match self.score_mode {
                    ScoreMode::Max => sorted_scores[0],
                    ScoreMode::Median => {
                        let trunc_mid = sorted_scores.len() / 2;
                        if sorted_scores.len() % 2 == 0 {
                            (sorted_scores[trunc_mid] + sorted_scores[trunc_mid - 1]) / 2.
                        } else {
                            sorted_scores[trunc_mid]
                        }
                    }
                    ScoreMode::Average => {
                        sorted_scores.iter().sum::<f32>() / sorted_scores.len() as f32
                    }
                };
                if score > threshold {
                    Some(RustpotterDetection {
                        name: wakeword.name.to_string(),
                        avg_score,
                        score,
                        scores,
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
        if self.partial_detection.is_some() && self.detection_countdown == 0 {
            let wakeword_detection = self.partial_detection.take().unwrap();
            if wakeword_detection.counter > self.min_scores {
                self.buffering = true;
                self.audio_features_window.clear();
                self.feature_extractor.reset();
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
                self.detection_countdown = self.max_features_frames / 2;
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
/// Encapsulates the detection information.
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct RustpotterDetection {
    /// Detected wakeword name.
    pub name: String,
    /// Detection score against the averaged template.
    pub avg_score: f32,
    /// Detection score.
    pub score: f32,
    /// Detection scores against each template.
    pub scores: HashMap<String, f32>,
    /// Partial detections counter.
    pub counter: usize,
}
