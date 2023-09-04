use std::collections::HashMap;

use crate::{
    audio::{AudioEncoder, BandPassFilter, GainNormalizerFilter},
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        MFCCS_EXTRACTOR_FRAME_SHIFT_MS, MFCCS_EXTRACTOR_PRE_EMPHASIS,
    },
    mfcc::{MfccExtractor, VadDetector},
    wakewords::{WakewordDetector, WakewordFile, WakewordV2},
    DetectorConfig, FiltersConfig, RustpotterConfig, Sample, ScoreMode, WakewordLoad,
    WakewordModel, WakewordRef,
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
/// rustpotter.add_wakeword_from_file("wakeword_key", "./tests/resources/oye_casa_g.rpw").unwrap();
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
    /// Required score against the averaged mfccs matrix. Only to discard frames.
    avg_threshold: f32,
    /// Required score while comparing the wakeword against the live stream.
    threshold: f32,
    /// Required number of partial scores (scores over threshold) to consider the detection real.
    min_scores: usize,
    /// Emit detection on min partial scores.
    eager: bool,
    /// How to calculate the final score.
    score_mode: ScoreMode,
    ///
    vad_detector: Option<VadDetector>,
    // Utils
    /// Utility to encode or re-encode the input wav data.
    wav_encoder: AudioEncoder,
    /// Utility to extract a collection of mfcc for each input audio frame.
    mfcc_extractor: MfccExtractor,
    /// Score reference for it to be expressed in a 0 - 1 range.
    score_ref: f32,
    /// Comparator band size.
    band_size: u16,
    /// Optional band-pass filter implementation.
    band_pass_filter: Option<BandPassFilter>,
    /// Optional gain filter implementation.
    gain_normalizer_filter: Option<GainNormalizerFilter>,
    // State
    /// The collection of active wakewords detectors.
    wakewords: HashMap<String, Box<dyn WakewordDetector>>,
    /// Indicates that the audio_mfcc_window has enough mfcc frames to run the detection.
    /// This means its length is greater or equal to the max_mfcc_frames value.
    buffering: bool,
    /// A window of mfccs frames extracted from the input audio.
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
    #[cfg(feature = "record")]
    // Path to create records
    record_path: Option<String>,
    #[cfg(feature = "record")]
    /// Audio data cache for recording
    audio_window: Vec<f32>,
    #[cfg(feature = "record")]
    /// Max audio data to retain
    max_audio_samples: usize,
}

impl Rustpotter {
    /// Returns a configured Rustpotter instance.
    pub fn new(config: &RustpotterConfig) -> Result<Rustpotter, String> {
        let reencoder = AudioEncoder::new(
            &config.fmt,
            MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )?;
        let samples_per_frame = reencoder.get_output_frame_length();
        let samples_per_shift = (samples_per_frame as f32
            / (MFCCS_EXTRACTOR_FRAME_LENGTH_MS as f32 / MFCCS_EXTRACTOR_FRAME_SHIFT_MS as f32))
            as usize;
        let mfcc_extractor = MfccExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
            samples_per_frame,
            samples_per_shift,
            0, // setup on wakeword added
            MFCCS_EXTRACTOR_PRE_EMPHASIS,
        );
        let gain_normalizer_filter = (&config.filters.gain_normalizer).into();
        let band_pass_filter = (&config.filters.band_pass).into();
        Ok(Rustpotter {
            avg_threshold: config.detector.avg_threshold,
            threshold: config.detector.threshold,
            min_scores: config.detector.min_scores,
            eager: config.detector.eager,
            score_mode: config.detector.score_mode,
            score_ref: config.detector.score_ref,
            band_size: config.detector.band_size,
            wav_encoder: reencoder,
            vad_detector: config.detector.vad_mode.map(VadDetector::new),
            mfcc_extractor,
            band_pass_filter,
            gain_normalizer_filter,
            audio_mfcc_window: Vec::new(),
            buffering: true,
            max_mfcc_frames: 0,
            wakewords: HashMap::new(),
            partial_detection: None,
            detection_countdown: 0,
            rms_level: 0.,
            gain: 1.,
            #[cfg(feature = "record")]
            max_audio_samples: 0,
            #[cfg(feature = "record")]
            audio_window: Vec::new(),
            #[cfg(feature = "record")]
            record_path: config.detector.record_path.clone(),
        })
    }
    /// Add wakeword ref to the detector.
    pub fn add_wakeword_ref(&mut self, key: &str, wakeword: WakewordRef) -> Result<(), String> {
        self.add_wakeword(key, wakeword)
    }
    /// Add wakeword model to the detector.
    pub fn add_wakeword_model(&mut self, key: &str, wakeword: WakewordModel) -> Result<(), String> {
        self.add_wakeword(key, wakeword)
    }
    /// Add wakeword from file bytes.
    pub fn add_wakeword_from_buffer(&mut self, key: &str, buffer: &[u8]) -> Result<(), String> {
        WakewordV2::load_from_buffer(buffer)
            .and_then(|w| self.add_wakeword_ref(key, w.into()))
            .or_else(|_| {
                WakewordRef::load_from_buffer(buffer)
                    .and_then(|wakeword| self.add_wakeword_ref(key, wakeword))
                    .or_else(|_| {
                        WakewordModel::load_from_buffer(buffer)
                            .and_then(|wakeword| self.add_wakeword_model(key, wakeword))
                    })
            })
    }
    /// Add wakeword from file path.
    pub fn add_wakeword_from_file(&mut self, key: &str, path: &str) -> Result<(), String> {
        WakewordV2::load_from_file(path)
            .and_then(|w| self.add_wakeword_ref(key, w.into()))
            .or_else(|_| {
                WakewordRef::load_from_file(path)
                    .and_then(|wakeword| self.add_wakeword_ref(key, wakeword))
                    .or_else(|_| {
                        WakewordModel::load_from_file(path)
                            .and_then(|wakeword| self.add_wakeword_model(key, wakeword))
                    })
            })
    }
    /// Remove wakeword by key.
    ///
    /// Returns true on success.
    pub fn remove_wakeword(&mut self, key: &str) -> bool {
        let len = self.wakewords.len();
        self.wakewords.retain(|k, _| !k.eq(key));
        if len != self.wakewords.len() {
            self.on_wakeword_change();
            true
        } else {
            false
        }
    }
    /// Remove all wakewords.
    ///
    /// Returns true on success.
    pub fn remove_wakewords(&mut self) -> bool {
        let len = self.wakewords.len();
        self.wakewords.clear();
        if len != self.wakewords.len() {
            self.on_wakeword_change();
            true
        } else {
            false
        }
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
    pub fn process_bytes(&mut self, audio_bytes: &[u8]) -> Option<RustpotterDetection> {
        if audio_bytes.len() != self.get_bytes_per_frame() {
            return None;
        }
        let encoded_samples = self.wav_encoder.encode_and_resample(audio_bytes);
        self.process_audio(encoded_samples)
    }
    /// Process encoded audio samples.
    ///
    /// Number of samples provided should match the return of the get_samples_per_frame method.
    ///
    pub fn process_samples<T: Sample>(
        &mut self,
        audio_samples: Vec<T>,
    ) -> Option<RustpotterDetection> {
        if audio_samples.len() != self.get_samples_per_frame() {
            return None;
        }
        let float_samples = self.wav_encoder.rencode_and_resample::<T>(audio_samples);
        self.process_audio(float_samples)
    }
    /// Updates detector and audio filters configs.
    ///
    pub fn update_config(&mut self, config: &RustpotterConfig) {
        self.update_detector_config(&config.detector);
        self.update_filters_config(&config.filters);
    }
    /// Updates detector config.
    ///
    pub fn update_detector_config(&mut self, config: &DetectorConfig) {
        self.avg_threshold = config.avg_threshold;
        self.threshold = config.threshold;
        self.min_scores = config.min_scores;
        self.eager = config.eager;
        self.band_size = config.band_size;
        self.score_ref = config.score_ref;
        self.score_mode = config.score_mode;
        self.vad_detector = config.vad_mode.map(VadDetector::new);
        #[cfg(feature = "record")]
        {
            self.record_path = config.record_path.clone();
        }
        for wd in self.wakewords.values_mut() {
            wd.update_config(self.score_ref, self.band_size, self.score_mode);
        }
        self.reset();
    }
    /// Updates audio filters config.
    ///
    pub fn update_filters_config(&mut self, config: &FiltersConfig) {
        self.band_pass_filter = (&config.band_pass).into();
        self.gain_normalizer_filter = (&config.gain_normalizer).into();
        self.reset();
    }
    /// Clean internal State
    ///
    pub fn reset(&mut self) {
        self.buffering = true;
        self.partial_detection = None;
        self.audio_mfcc_window.clear();
        self.mfcc_extractor.reset();
        if let Some(vad) = self.vad_detector.as_mut() {
            vad.reset();
        }
        #[cfg(feature = "record")]
        {
            self.audio_window.clear();
        }
    }

    fn add_wakeword<T: WakewordFile>(&mut self, name: &str, wakeword: T) -> Result<(), String> {
        if self.wakewords.is_empty() {
            self.reset();
            self.mfcc_extractor.set_out_size(wakeword.get_mfcc_size());
        } else if !self
            .wakewords
            .values()
            .next()
            .unwrap()
            .get_mfcc_size()
            .eq(&wakeword.get_mfcc_size())
        {
            return Err(
                "Usage of wakewords with different mfcc size is not supported, ignoring wakeword"
                    .to_string(),
            );
        }
        self.wakewords.insert(
            name.to_string(),
            wakeword.get_detector(self.score_ref, self.band_size, self.score_mode),
        );
        self.on_wakeword_change();
        Ok(())
    }
    fn on_wakeword_change(&mut self) {
        let mut max_mfcc_frames = usize::MIN;
        let mut target_rms_level = f32::NAN;
        for wakeword in self.wakewords.values() {
            max_mfcc_frames = wakeword.as_ref().get_mfcc_frame_size().max(max_mfcc_frames);
            target_rms_level = wakeword.get_rms_level().max(target_rms_level);
        }
        self.max_mfcc_frames = max_mfcc_frames;
        if let Some(gain_normalizer_filter) = self.gain_normalizer_filter.as_mut() {
            gain_normalizer_filter.set_rms_level_ref(target_rms_level, self.max_mfcc_frames / 3);
        }
        self.buffering = self.audio_mfcc_window.len() < self.max_mfcc_frames;
        #[cfg(feature = "record")]
        if self.record_path.is_some() {
            self.max_audio_samples = (self.max_mfcc_frames
                / crate::constants::MFCCS_EXTRACTOR_OUT_SHIFTS)
                * self.wav_encoder.get_output_frame_length();
        }
    }
    fn process_audio(&mut self, mut audio_buffer: Vec<f32>) -> Option<RustpotterDetection> {
        if self.wakewords.is_empty() {
            return None;
        }
        #[cfg(feature = "record")]
        if self.record_path.is_some() {
            self.audio_window.append(&mut (audio_buffer.clone()));
            if self.audio_window.len() > self.max_audio_samples {
                self.audio_window.drain(0..audio_buffer.len());
            }
        }
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
    fn process_new_mfccs(&mut self, mfcc_frame: Vec<f32>) -> Option<RustpotterDetection> {
        let mut result: Option<RustpotterDetection> = None;
        let should_run = self.partial_detection.is_some()
            || self
                .vad_detector
                .as_mut()
                .map_or(true, |v| v.is_voice(&mfcc_frame));
        self.audio_mfcc_window.push(mfcc_frame);
        if self.audio_mfcc_window.len() >= self.max_mfcc_frames {
            if self.buffering {
                self.buffering = false;
            }
            if should_run {
                result = self.run_detection();
            }
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
        if self.partial_detection.is_some()
            && self.is_detection_done(self.partial_detection.as_ref().unwrap())
        {
            let wakeword_detection = self.partial_detection.take().unwrap();
            if wakeword_detection.counter >= self.min_scores {
                self.reset();
                return Some(wakeword_detection);
            }
        }
        let wakeword_detection = self.run_wakeword_detectors().map(|mut detection| {
            detection.counter = self.partial_detection.as_ref().map_or(1, |d| d.counter + 1);
            detection.gain = self.gain;
            detection
        });
        if let Some(wakeword_detection) = wakeword_detection {
            if self.partial_detection.is_none()
                || self.partial_detection.as_ref().unwrap().score < wakeword_detection.score
            {
                #[cfg(feature = "record")]
                if let Some(record_path) = self.record_path.as_ref() {
                    self.create_audio_record(record_path, &wakeword_detection);
                }
                self.partial_detection = Some(wakeword_detection);
            } else {
                let partial_detection = self.partial_detection.as_mut().unwrap();
                partial_detection.counter = wakeword_detection.counter;
            }
            self.detection_countdown = self.max_mfcc_frames / 2;
        }
        None
    }
    fn run_wakeword_detectors(&mut self) -> Option<RustpotterDetection> {
        let mut wakeword_detections = self
            .wakewords
            .values()
            .filter_map(|wakeword| {
                wakeword.run_detection(
                    self.audio_mfcc_window.clone(),
                    self.avg_threshold,
                    self.threshold,
                )
            })
            .collect::<Vec<RustpotterDetection>>();
        wakeword_detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        wakeword_detections.into_iter().next()
    }
    fn is_detection_done(&self, detection: &RustpotterDetection) -> bool {
        if self.detection_countdown == 0 {
            true
        } else {
            self.eager && detection.counter >= self.min_scores
        }
    }
    #[cfg(feature = "record")]
    fn create_audio_record(&self, record_path: &str, detection: &RustpotterDetection) {
        let spec = hound::WavSpec {
            sample_rate: DETECTOR_INTERNAL_SAMPLE_RATE as u32,
            sample_format: hound::SampleFormat::Float,
            bits_per_sample: 32,
            channels: 1,
        };
        let timestamp = std::time::UNIX_EPOCH.elapsed().unwrap().as_millis();
        let record_folder = std::path::Path::new(record_path);
        if !record_folder.exists() {
            return;
        }
        let file_path = record_folder.join(
            "[".to_string()
                + &detection.name
                + "]"
                + timestamp.to_string().as_str()
                + "-"
                + &detection.score.to_string().replace(".", "_")
                + ".wav",
        );
        let writer = hound::WavWriter::create(file_path.as_os_str(), spec);
        if let Ok(mut writer) = writer {
            self.audio_window
                .iter()
                .for_each(|sample| _ = writer.write_sample(*sample));
            _ = writer.flush();
        }
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
