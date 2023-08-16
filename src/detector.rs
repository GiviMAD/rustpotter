use std::collections::HashMap;

use crate::{
    audio::{BandPassFilter, GainNormalizerFilter, WAVEncoder},
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        MFCCS_EXTRACTOR_FRAME_SHIFT_MS, MFCCS_EXTRACTOR_NUM_COEFFICIENT,
        MFCCS_EXTRACTOR_PRE_EMPHASIS,
    },
    mfcc::{MfccComparator, MfccExtractor},
    wakewords::WakewordDetector,
    DeserializableWakeword, RustpotterConfig, ScoreMode, WakewordModel, WakewordRef,
    SampleType,
};
/// Rustpotter is an open source wakeword spotter forged in rust
/// ```
/// use rustpotter::{Rustpotter, RustpotterConfig};
/// // assuming the audio input format match the rustpotter defaults
/// let mut rustpotter_config = RustpotterConfig::default();
/// // Configure the rustpotter
/// // ...
/// let mut rustpotter = Rustpotter::new(&rustpotter_config).unwrap();
/// // load and enable a wakeword
/// rustpotter.add_wakeword_from_file("./tests/resources/oye_casa_g.rpw").unwrap();
/// // You need a buffer of size rustpotter.get_samples_per_frame() when using samples or rustpotter.get_bytes_per_frame() when using bytes.  
/// let mut sample_buffer: Vec<i16> = vec![0; rustpotter.get_samples_per_frame()];
/// // while true { Iterate forever
///     // fill the buffer with the required samples/bytes...
///    let detection = rustpotter.process_samples(sample_buffer);
///     if let Some(detection) = detection {
///         println!("{:?}", detection);
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
    /// Utility to extract a collection of mfcc for each input audio frame.
    mfcc_extractor: MfccExtractor,
    /// Utility to measure the similarity between two feature frame vectors.
    mfcc_comparator: MfccComparator,
    /// Optional band-pass filter implementation.
    band_pass_filter: Option<BandPassFilter>,
    /// Optional gain filter implementation.
    gain_normalizer_filter: Option<GainNormalizerFilter>,
    // State
    /// The collection of active wakewords detectors.
    wakewords: Vec<Box<dyn WakewordDetector>>,
    /// Indicates that the audio_mfcc_window has enough mfcc frames to run the detection.
    /// This means its length is greater or equal to the max_mfcc_frames value.
    buffering: bool,
    /// A window of feature frames extract from the input audio.
    /// It grows until match the max_mfcc_frames value.
    audio_mfcc_window: Vec<Vec<f32>>,
    /// Max number of feature frames on the wakewords.
    max_mfcc_frames: usize,
    /// Stores the partial detection with greater score while waiting for the detection countdown to be zero.
    partial_detection: Option<RustpotterDetection>,
    /// Countdown until detection fires.
    /// Whenever a better partial detection is found, it is set to the double of the max number of feature frames in the wakeword samples.
    /// When it gets to zero and a partial detection exists, it'll be considered the final detection.
    detection_countdown: usize,
    /// Current frame rms level
    rms_level: f32,
    /// Gain normalization applied to current frame.
    gain: f32,
}

impl Rustpotter {
    /// Returns a configured Rustpotter instance.
    pub fn new(config: &RustpotterConfig) -> Result<Rustpotter, String> {
        let reencoder = WAVEncoder::new(
            &config.fmt,
            MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )?;
        let samples_per_frame = reencoder.get_output_frame_length();
        let samples_per_shift = (samples_per_frame as f32
            / (MFCCS_EXTRACTOR_FRAME_LENGTH_MS as f32 / MFCCS_EXTRACTOR_FRAME_SHIFT_MS as f32))
            as usize;
        let feature_extractor = MfccExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
            samples_per_frame,
            samples_per_shift,
            MFCCS_EXTRACTOR_NUM_COEFFICIENT,
            MFCCS_EXTRACTOR_PRE_EMPHASIS,
        );
        let mfcc_comparator = MfccComparator::new(
            config.detector.score_ref,
        );
        let band_pass_filter = if config.filters.band_pass.enabled {
            Some(BandPassFilter::new(
                DETECTOR_INTERNAL_SAMPLE_RATE as f32,
                config.filters.band_pass.low_cutoff,
                config.filters.band_pass.high_cutoff,
            ))
        } else {
            None
        };
        let gain_normalizer_filter = if config.filters.gain_normalizer.enabled {
            Some(GainNormalizerFilter::new(
                config.filters.gain_normalizer.min_gain,
                config.filters.gain_normalizer.max_gain,
                config.filters.gain_normalizer.gain_ref,
            ))
        } else {
            None
        };
        Ok(Rustpotter {
            avg_threshold: config.detector.avg_threshold,
            threshold: config.detector.threshold,
            min_scores: config.detector.min_scores,
            score_mode: config.detector.score_mode,
            wav_encoder: reencoder,
            mfcc_extractor: feature_extractor,
            mfcc_comparator,
            band_pass_filter,
            gain_normalizer_filter,
            audio_mfcc_window: Vec::new(),
            buffering: true,
            max_mfcc_frames: 0,
            wakewords: Vec::new(),
            partial_detection: None,
            detection_countdown: 0,
            rms_level: 0.,
            gain: 1.,
        })
    }
    /// Add wakeword to the detector.
    pub fn add_wakeword(&mut self, wakeword: WakewordRef) {
        self.wakewords.push(Box::new(
            wakeword.get_comparator(self.mfcc_comparator.clone(), self.score_mode),
        ));
        self.on_wakeword_change();
    }
    /// Add wakeword model to the detector.
    pub fn add_wakeword_model(&mut self, wakeword: WakewordModel) {
        let score_ref = self.mfcc_comparator.get_score_ref();
        self.wakewords.push(Box::new(wakeword.get_nn(score_ref)));
        self.on_wakeword_change();
    }
    /// Remove wakeword by name or label.
    pub fn remove_wakeword(&mut self, name: &str) -> bool {
        let len = self.wakewords.len();
        self.wakewords.retain(|w| !w.contains(name));
        if len != self.wakewords.len() {
            self.on_wakeword_change();
            true
        } else {
            false
        }
    }
    /// Update detection window and gain normalizer requirements.
    fn on_wakeword_change(&mut self) {
        let mut max_feature_frames = usize::MIN;
        let mut target_rms_level = f32::NAN;
        for wakeword in self.wakewords.iter() {
            max_feature_frames = wakeword
                .as_ref()
                .get_mfcc_frame_size()
                .max(max_feature_frames);
            target_rms_level = wakeword.get_rms_level().max(target_rms_level);
        }
        self.max_mfcc_frames = max_feature_frames;
        if let Some(gain_normalizer_filter) = self.gain_normalizer_filter.as_mut() {
            gain_normalizer_filter.set_rms_level_ref(target_rms_level, self.max_mfcc_frames / 3);
        }
        self.buffering = self.audio_mfcc_window.len() < self.max_mfcc_frames;
    }
    /// Add wakeword from model bytes.
    pub fn add_wakeword_from_buffer(&mut self, buffer: &[u8]) -> Result<(), String> {
        WakewordRef::load_from_buffer(buffer)
            .map(|wakeword| self.add_wakeword(wakeword))
            .or_else(|_| {
                WakewordModel::load_from_buffer(buffer)
                    .map(|wakeword| self.add_wakeword_model(wakeword))
            })
    }
    /// Add wakeword from model path.
    pub fn add_wakeword_from_file(&mut self, path: &str) -> Result<(), String> {
        WakewordRef::load_from_file(path)
            .map(|wakeword| self.add_wakeword(wakeword))
            .or_else(|_| {
                WakewordModel::load_from_file(path)
                    .map(|wakeword| self.add_wakeword_model(wakeword))
            })
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
    /// Returns the gain applied to the latest frame by the gain normalizer filter (1. if none or disabled).
    pub fn get_gain(&self) -> f32 {
        self.gain
    }
    /// Returns the gain normalizer filter rms level reference.
    pub fn get_rms_level_ref(&self) -> f32 {
        self.gain_normalizer_filter
            .as_ref()
            .map(|f| f.get_rms_level_ref())
            .unwrap_or(f32::NAN)
    }
    /// Process bytes buffer.
    ///
    /// Number of bytes provided should match the return of the get_bytes_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Assumes buffer endianness matches the configured for the detector.
    ///
    pub fn process_bytes(&mut self, audio_bytes: &[u8]) -> Option<RustpotterDetection> {
        let encoded_samples = self.wav_encoder.encode_and_resample(audio_bytes);
        self.process_audio(encoded_samples)
    }
    /// Process encoded audio samples.
    ///
    /// Number of samples provided should match the return of the get_samples_per_frame method.
    ///
    pub fn process_samples<T: SampleType>(
        &mut self,
        audio_samples: Vec<T>,
    ) -> Option<RustpotterDetection> {
        let float_samples = self.wav_encoder.rencode_and_resample::<T>(audio_samples);
        self.process_audio(float_samples)
    }
    fn process_audio(&mut self, mut audio_buffer: Vec<f32>) -> Option<RustpotterDetection> {
        self.rms_level = GainNormalizerFilter::get_rms_level(&audio_buffer);
        if self.gain_normalizer_filter.is_some() {
            self.gain = self
                .gain_normalizer_filter
                .as_mut()
                .unwrap()
                .filter(&mut audio_buffer, self.rms_level);
        }
        if self.band_pass_filter.is_some() {
            self.band_pass_filter
                .as_mut()
                .unwrap()
                .filter(&mut audio_buffer);
        }
        self.mfcc_extractor
            .compute(&audio_buffer)
            .into_iter()
            .find_map(|mfccs| self.process_new_mfccs(mfccs))
    }
    fn process_new_mfccs(&mut self, features_frame: Vec<f32>) -> Option<RustpotterDetection> {
        let mut result: Option<RustpotterDetection> = None;
        self.audio_mfcc_window.push(features_frame);
        if self.audio_mfcc_window.len() >= self.max_mfcc_frames {
            if self.buffering {
                self.buffering = false;
            }
            result = self.run_detection();
        }
        if self.audio_mfcc_window.len() >= self.max_mfcc_frames {
            self.audio_mfcc_window.drain(0..1);
        }
        result
    }
    fn run_detection(&mut self) -> Option<RustpotterDetection> {
        if self.detection_countdown != 0 {
            self.detection_countdown -= 1;
        }
        let wakeword_detection = self.run_wakeword_detectors().map(|mut detection| {
            detection.counter = self.partial_detection.as_ref().map_or(1, |d| d.counter + 1);
            detection.gain = self.gain;
            detection
        });
        if self.partial_detection.is_some() && self.detection_countdown == 0 {
            let wakeword_detection = self.partial_detection.take().unwrap();
            if wakeword_detection.counter >= self.min_scores {
                self.buffering = true;
                self.audio_mfcc_window.clear();
                self.mfcc_extractor.reset();
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
                    let partial_detection = self.partial_detection.as_mut().unwrap();
                    partial_detection.counter = wakeword_detection.as_ref().unwrap().counter;
                    partial_detection.gain = self.gain;
                }
                self.detection_countdown = self.max_mfcc_frames / 2;
            }
            None
        }
    }

    fn run_wakeword_detectors(&mut self) -> Option<RustpotterDetection> {
        let mut wakeword_detections = self
            .wakewords
            .iter()
            .filter_map(|wakeword| {
                wakeword.run_detection(
                    self.audio_mfcc_window.to_vec(),
                    self.avg_threshold,
                    self.threshold,
                )
            })
            .collect::<Vec<RustpotterDetection>>();
        wakeword_detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        wakeword_detections.into_iter().next()
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
    /// Gain applied to the scored frame
    pub gain: f32,
}
