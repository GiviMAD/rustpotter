use crate::comparator::FeatureComparator;
use crate::nnnoiseless_fork::{self, DenoiseFeatures};
use crate::wakeword::{Wakeword, WakewordModel};
use hound::WavReader;
use log::{debug, warn};
use rubato::{FftFixedInOut, Resampler};
use savefile::{load_file, load_from_mem, save_file, save_to_mem};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::SystemTime;
static INTERNAL_SAMPLE_RATE: usize = 48000;
/// Allowed vad modes
pub type VadMode = webrtc_vad::VadMode;
/// Allowed wav sample formats
pub type SampleFormat = hound::SampleFormat;
/// Use this struct to configure and build your wakeword detector.
/// ```
/// let mut word_detector = detector_builder
///     .set_threshold(0.5)
///     .set_sample_rate(16000)
///     .set_eager_mode(true)
///     .set_single_thread(true)
///     .build();
/// ```
pub struct WakewordDetectorBuilder {
    sample_rate: Option<usize>,
    sample_format: Option<SampleFormat>,
    bits_per_sample: Option<u16>,
    vad_mode: Option<VadMode>,
    vad_sensitivity: Option<f32>,
    vad_delay: Option<u16>,
    eager_mode: bool,
    single_thread: bool,
    averaged_threshold: Option<f32>,
    threshold: Option<f32>,
    comparator_band_size: Option<usize>,
    comparator_ref: Option<f32>,
}
impl WakewordDetectorBuilder {
    pub fn new() -> Self {
        WakewordDetectorBuilder {
            // input options
            sample_rate: None,
            sample_format: None,
            bits_per_sample: None,
            // detection options
            eager_mode: false,
            single_thread: false,
            vad_mode: None,
            vad_delay: None,
            vad_sensitivity: None,
            threshold: None,
            averaged_threshold: None,
            comparator_band_size: None,
            comparator_ref: None,
        }
    }
    pub fn build(&self) -> WakewordDetector {
        WakewordDetector::new(
            self.get_sample_rate(),
            self.get_sample_format(),
            self.get_bits_per_sample(),
            self.get_eager_mode(),
            self.get_single_thread(),
            self.get_vad_mode(),
            self.get_vad_delay(),
            self.get_vad_sensitivity(),
            self.get_threshold(),
            self.get_averaged_threshold(),
            self.get_comparator_band_size(),
            self.get_comparator_ref(),
        )
    }
    /// Configures the detector threshold,
    /// is the min score (in range 0. to 1.) that some of
    /// the wakeword templates should obtain to trigger a detection.
    ///
    /// Defaults to 0.5, wakeword defined value takes prevalence if present.
    pub fn set_threshold(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.threshold = Some(value);
        self
    }
    /// Configures the detector threshold,
    /// is the min score (in range 0. to 1.) that  
    /// the averaged wakeword template should obtain to allow
    /// to continue with the detection. This way it can prevent to
    /// run the comparison of the current frame against each of the wakeword templates.
    /// If set to 0. this functionality is disabled.
    ///
    /// Defaults to half of the configured threshold, wakeword defined value takes prevalence if present.
    pub fn set_averaged_threshold(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.averaged_threshold = Some(value);
        self
    }
    /// Configures the detector expected bit per sample for the audio chunks to process.
    ///
    /// Defaults to 16; Allowed values: 8, 16, 24, 32
    pub fn set_bits_per_sample(&mut self, value: u16) -> &mut Self {
        self.bits_per_sample = Some(value);
        self
    }
    /// Configures the detector expected sample rate for the audio chunks to process.
    ///
    /// Defaults to 16000
    pub fn set_sample_rate(&mut self, value: usize) -> &mut Self {
        self.sample_rate = Some(value);
        self
    }
    /// Configures the detector expected sample format for the audio chunks to process.
    ///
    /// Defaults to int
    pub fn set_sample_format(&mut self, value: SampleFormat) -> &mut Self {
        self.sample_format = Some(value);
        self
    }
    /// Configures the band-size for the comparator used to match the samples.
    ///
    /// Defaults to 6
    pub fn set_comparator_band_size(&mut self, value: usize) -> &mut Self {
        self.comparator_band_size = Some(value);
        self
    }
    /// Configures the reference for the comparator used to match the samples.
    ///
    /// Defaults to 0.22
    pub fn set_comparator_ref(&mut self, value: f32) -> &mut Self {
        self.comparator_ref = Some(value);
        self
    }
    /// Enables eager mode.
    /// Terminate the detection as son as one result is above the score,
    /// instead of wait to see if the next frame has a higher score.
    ///
    /// Recommended for real usage.
    ///
    /// Defaults to false.
    pub fn set_eager_mode(&mut self, value: bool) -> &mut Self {
        self.eager_mode = value;
        self
    }
    /// Unless enabled the comparison against multiple wakewords run
    /// in separate threads.
    ///
    /// Defaults to false.
    ///
    /// Only applies when more than a wakeword is loaded.
    pub fn set_single_thread(&mut self, value: bool) -> &mut Self {
        self.single_thread = value;
        self
    }
    /// Seconds to disable the vad detector after voice is detected.
    ///
    /// Defaults to 3.
    ///
    /// Only applies if vad is enabled.
    pub fn set_vad_delay(&mut self, value: u16) -> &mut Self {
        self.vad_delay = Some(value);
        self
    }
    /// Voice/silence ratio in the last second to consider voice detected.
    ///
    /// Defaults to 0.5.
    ///
    /// Only applies if vad is enabled.
    pub fn set_vad_sensitivity(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.vad_sensitivity = Some(value);
        self
    }
    /// Use a vad detector to reduce computation on absence of voice sound.
    ///
    /// Unless specified the vad detector is disabled.
    pub fn set_vad_mode(&mut self, value: VadMode) -> &mut Self {
        self.vad_mode = Some(value);
        self
    }
    fn get_threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5)
    }
    fn get_averaged_threshold(&self) -> f32 {
        self.averaged_threshold.unwrap_or(self.get_threshold() / 2.)
    }
    fn get_sample_rate(&self) -> usize {
        self.sample_rate.unwrap_or(INTERNAL_SAMPLE_RATE)
    }
    fn get_sample_format(&self) -> SampleFormat {
        self.sample_format.unwrap_or(SampleFormat::Int)
    }
    fn get_bits_per_sample(&self) -> u16 {
        self.bits_per_sample.unwrap_or(16)
    }
    fn get_comparator_band_size(&self) -> usize {
        self.comparator_band_size.unwrap_or(11)
    }
    fn get_comparator_ref(&self) -> f32 {
        self.comparator_ref.unwrap_or(0.22)
    }
    fn get_eager_mode(&self) -> bool {
        self.eager_mode
    }
    fn get_single_thread(&self) -> bool {
        self.single_thread
    }
    fn get_vad_sensitivity(&self) -> f32 {
        self.vad_sensitivity.unwrap_or(0.5)
    }
    fn get_vad_delay(&self) -> u16 {
        self.vad_delay.unwrap_or(3)
    }
    fn get_vad_mode(&self) -> Option<VadMode> {
        if self.vad_mode.is_none() {
            return None;
        }
        match self.vad_mode.as_ref().unwrap() {
            VadMode::Quality => Some(VadMode::Quality),
            VadMode::LowBitrate => Some(VadMode::LowBitrate),
            VadMode::Aggressive => Some(VadMode::Aggressive),
            VadMode::VeryAggressive => Some(VadMode::VeryAggressive),
        }
    }
}
/// This struct manages the wakeword generation and the spotting functionality.
///
/// ```
/// // assuming the audio input format match the detector defaults
/// let mut word_detector = detector_builder.build();
/// // load and enable a wakeword
/// word_detector.add_keyword_from_model("./model.rpw", true)?;
/// let mut frame_buffer: Vec<i32> = vec![0; word_detector.get_samples_per_frame()];
/// while true {
///     // fill the buffer with new samples...
///     let detection = word_detector.process(frame_buffer);
///     if detection.is_some() {
///         println!(
///             "Detected '{}' with score {}!",
///             detection.unwrap().wakeword,
///             detection.unwrap().score,
///         );
///     }
/// }
/// ```
pub struct WakewordDetector {
    // input options
    sample_format: SampleFormat,
    bits_per_sample: u16,
    // detection options
    threshold: f32,
    averaged_threshold: f32,
    eager_mode: bool,
    single_thread: bool,
    resampler: Option<FftFixedInOut<f32>>,
    comparator: Arc<FeatureComparator>,
    vad_detector: Option<webrtc_vad::Vad>,
    vad_delay: u16,
    vad_sensitivity: f32,
    // state
    samples_per_frame: usize,
    buffering: bool,
    min_frames: usize,
    max_frames: usize,
    frames: Vec<Vec<f32>>,
    wakewords: HashMap<String, Wakeword>,
    result_state: Option<DetectedWakeword>,
    extractor: DenoiseFeatures,
    resampler_out_buffer: Option<Vec<Vec<f32>>>,
    vad_enabled: bool,
    voice_detections: Vec<bool>,
    voice_detection_time: SystemTime,
    audio_cache: Vec<Vec<f32>>,
}
impl WakewordDetector {
    /// Creates a new WakewordDetector.
    ///
    /// It is recommended to use the WakewordDetectorBuilder struct instead.
    pub fn new(
        // input options
        sample_rate: usize,
        sample_format: SampleFormat,
        bits_per_sample: u16,
        // detection options
        eager_mode: bool,
        single_thread: bool,
        vad_mode: Option<VadMode>,
        vad_delay: u16,
        vad_sensitivity: f32,
        threshold: f32,
        averaged_threshold: f32,
        comparator_band_size: usize,
        comparator_ref: f32,
    ) -> Self {
        let mut samples_per_frame = 480;
        let resampler = if sample_rate != INTERNAL_SAMPLE_RATE {
            let resampler =
                FftFixedInOut::<f32>::new(sample_rate, INTERNAL_SAMPLE_RATE, samples_per_frame, 1)
                    .unwrap();
            samples_per_frame = resampler.input_frames_next();
            Some(resampler)
        } else {
            None
        };
        let vad_detector = if vad_mode.is_some() {
            Some(webrtc_vad::Vad::new_with_rate_and_mode(
                webrtc_vad::SampleRate::Rate48kHz,
                vad_mode.unwrap(),
            ))
        } else {
            None
        };
        let detector = WakewordDetector {
            threshold,
            averaged_threshold,
            sample_format,
            bits_per_sample,
            samples_per_frame,
            frames: Vec::new(),
            wakewords: HashMap::new(),
            buffering: true,
            min_frames: 9999,
            max_frames: 0,
            result_state: None,
            extractor: DenoiseFeatures::new(),
            comparator: Arc::new(FeatureComparator::new(comparator_band_size, comparator_ref)),
            resampler_out_buffer: if resampler.is_some() {
                Some(resampler.as_ref().unwrap().output_buffer_allocate())
            } else {
                None
            },
            resampler,
            eager_mode,
            single_thread,
            voice_detections: Vec::with_capacity(100),
            audio_cache: Vec::with_capacity(100),
            vad_enabled: false,
            vad_detector,
            vad_delay,
            vad_sensitivity,
            voice_detection_time: SystemTime::UNIX_EPOCH,
        };
        detector
    }
    /// Loads a wakeword from its model bytes.
    pub fn add_keyword_from_model_bytes(
        &mut self,
        bytes: Vec<u8>,
        enabled: bool,
    ) -> Result<(), String> {
        let model: WakewordModel = load_from_mem(&bytes, 0).or(Err("Unable to load model data"))?;
        self.add_keyword_from_model(model, enabled)
    }
    /// Loads a wakeword from its model path.
    pub fn add_keyword_from_model_file(
        &mut self,
        path: String,
        enabled: bool,
    ) -> Result<(), String> {
        let model: WakewordModel = load_file(path, 0).or(Err("Unable to load model data"))?;
        self.add_keyword_from_model(model, enabled)
    }
    /// Generates the model file bytes from a loaded a wakeword
    pub fn generate_wakeword_model_bytes(&self, name: String) -> Result<Vec<u8>, String> {
        let model = self.get_wakeword_model(&name)?;
        save_to_mem(0, &model).or(Err(String::from("Unable to generate model bytes")))
    }
    /// Generates a model file from a loaded a wakeword on the desired path
    pub fn generate_wakeword_model_file(&self, name: String, path: String) -> Result<(), String> {
        let model = self.get_wakeword_model(&name)?;
        save_file(path, 0, &model).or(Err(String::from("Unable to generate file")))
    }
    /// Adds a wakeword using wav samples.
    pub fn add_keyword(
        &mut self,
        name: String,
        enabled: bool,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
        sample_paths: Vec<String>,
    ) {
        debug!(
            "Adding keyword \"{}\" (sample paths: {:?})",
            name, sample_paths
        );
        if self.wakewords.get_mut(&name).is_none() {
            self.wakewords.insert(
                name.clone(),
                Wakeword::new(enabled, averaged_threshold, threshold),
            );
        }
        let mut min_frames: usize = 0;
        let mut max_frames: usize = 0;
        for template in sample_paths {
            match self.extract_features_from_file(template) {
                Ok(features) => {
                    let word = self.wakewords.get_mut(&name).unwrap();
                    word.add_features(features.to_vec());
                    min_frames = if min_frames == 0 {
                        word.get_min_frames()
                    } else {
                        std::cmp::min(min_frames, features.len())
                    };
                    max_frames = std::cmp::min(max_frames, features.len());
                }
                Err(msg) => {
                    warn!("{}", msg);
                }
            };
        }
        self.update_detection_frame_size(min_frames, max_frames);
    }
    /// Process i32 audio chunks.
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    /// Assumes sample rate match the configured for the detector.
    /// Asserts that detector bits_per_sample is one of: 8, 16, 24, 32.
    /// Asserts that detector sample_format is 'int'.
    /// It's an alias for the process_i32 method.
    pub fn process(&mut self, audio_chunk: &[i32]) -> Option<DetectedWakeword> {
        self.process_i32(audio_chunk)
    }
    /// Process i8 audio chunks.
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    /// Assumes sample rate match the configured for the detector.
    /// Asserts that detector bits_per_sample is 8.
    /// Asserts that detector sample_format is 'int'.
    pub fn process_i8(&mut self, audio_chunk: &[i8]) -> Option<DetectedWakeword> {
        assert!(self.bits_per_sample == 8);
        self.process_int(
            &audio_chunk
                .into_iter()
                .map(|i| *i as i32)
                .collect::<Vec<_>>(),
        )
    }
    /// Process i16 audio chunks.
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    /// Assumes sample rate match the configured for the detector.
    /// Asserts that detector bits_per_sample is one of: 8, 16.
    /// Asserts that detector sample_format is 'int'.
    pub fn process_i16(&mut self, audio_chunk: &[i16]) -> Option<DetectedWakeword> {
        assert!(self.bits_per_sample == 8 || self.bits_per_sample == 16);
        self.process_int(
            &audio_chunk
                .into_iter()
                .map(|i| *i as i32)
                .collect::<Vec<_>>(),
        )
    }
    /// Process i32 audio chunks.
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    /// Assumes sample rate match the configured for the detector.
    /// Asserts that detector bits_per_sample is one of: 8, 16, 24, 32.
    /// Asserts that detector sample_format is 'int'.
    pub fn process_i32(&mut self, audio_chunk: &[i32]) -> Option<DetectedWakeword> {
        assert!(
            self.bits_per_sample == 8
                || self.bits_per_sample == 16
                || self.bits_per_sample == 24
                || self.bits_per_sample == 32
        );
        self.process_int(audio_chunk)
    }
    /// Process f32 audio chunks.
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    /// Assumes sample rate match the configured for the detector.
    /// Asserts that detector bits_per_sample is 32.
    /// Asserts that detector sample_format is 'float'.
    pub fn process_f32(&mut self, audio_chunk: &[f32]) -> Option<DetectedWakeword> {
        assert!(audio_chunk.len() == self.samples_per_frame);
        assert!(self.bits_per_sample == 32);
        assert!(self.sample_format == SampleFormat::Float);
        let float_buffer: Vec<f32> = audio_chunk
            .into_iter()
            .map(|s| s * 32767.0)
            .collect::<Vec<_>>();
        let resampled_audio = if self.resampler.is_some() {
            let resampler = self.resampler.as_mut().unwrap();
            let waves_in = vec![float_buffer; 1];
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            resampler
                .process_into_buffer(&waves_in, waves_out, None)
                .unwrap();
            let result = waves_out.get(0).unwrap();
            result.to_vec()
        } else {
            float_buffer
        };
        self.apply_vad_detection(resampled_audio)
    }
    /// Returns the desired chunk size.
    pub fn get_samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }
    fn add_keyword_from_model(
        &mut self,
        model: WakewordModel,
        enabled: bool,
    ) -> Result<(), String> {
        let keyword = model.keyword.clone();
        let wakeword = Wakeword::from_model(model, enabled);
        self.update_detection_frame_size(wakeword.get_min_frames(), wakeword.get_max_frames());
        self.wakewords.insert(keyword, wakeword);
        Ok(())
    }
    fn update_detection_frame_size(&mut self, min_frames: usize, max_frames: usize) {
        self.min_frames = std::cmp::min(self.min_frames, min_frames);
        self.max_frames = std::cmp::max(self.max_frames, max_frames);
    }
    fn get_wakeword_model(&self, name: &String) -> Result<WakewordModel, String> {
        let wakeword = self.wakewords.get(name);
        if wakeword.is_none() {
            Err(String::from("Missing wakeword"))
        } else {
            let features = wakeword.unwrap().get_templates();
            let model = WakewordModel::new(
                name.clone(),
                features,
                wakeword.unwrap().get_averaged_threshold(),
                wakeword.unwrap().get_threshold(),
            );
            Ok(model)
        }
    }
    fn process_int(&mut self, audio_chunk: &[i32]) -> Option<DetectedWakeword> {
        assert!(audio_chunk.len() == self.samples_per_frame);
        assert!(self.sample_format == SampleFormat::Int);
        let resampled_audio = if self.resampler.is_some() {
            let resampler = self.resampler.as_mut().unwrap();
            let bits_per_sample = self.bits_per_sample;
            let float_buffer: Vec<f32> = audio_chunk
                .into_iter()
                .map(move |s| {
                    if bits_per_sample < 16 {
                        (*s << (16 - bits_per_sample)) as f32
                    } else {
                        (*s >> (bits_per_sample - 16)) as f32
                    }
                })
                .collect::<Vec<_>>();
            let waves_in = vec![float_buffer; 1];
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            resampler
                .process_into_buffer(&waves_in, waves_out, None)
                .unwrap();
            let result = waves_out.get(0).unwrap();
            result.to_vec()
        } else {
            audio_chunk
                .into_iter()
                .map(|n| *n as f32)
                .collect::<Vec<_>>()
        };
        self.apply_vad_detection(resampled_audio)
    }
    fn apply_vad_detection(&mut self, resampled_audio: Vec<f32>) -> Option<DetectedWakeword> {
        if self.result_state.is_none()
            && self.vad_detector.is_some()
            && (self.vad_enabled
                || self.voice_detection_time.elapsed().unwrap().as_secs() >= self.vad_delay as u64)
        {
            let vad = self.vad_detector.as_mut().unwrap();
            if !self.vad_enabled {
                debug!("switching to vad detector");
                self.vad_enabled = true;
            }
            let is_voice_result = vad.is_voice_segment(
                &resampled_audio
                    .iter()
                    .map(|i| *i as i16)
                    .collect::<Vec<i16>>(),
            );
            self.voice_detections
                .push(is_voice_result.is_err() || is_voice_result.unwrap());
            if self.voice_detections.len() < 30 {
                return self.process_encoded_audio(&resampled_audio, true);
            }
            if self.voice_detections.len() > 100 {
                self.voice_detections.drain(0..1);
            }
            if self.voice_detections.iter().filter(|i| **i == true).count()
                >= (self.vad_sensitivity * self.voice_detections.len() as f32) as usize
            {
                debug!("voice detected; processing cache");
                self.audio_cache
                    .drain(0..self.audio_cache.len())
                    .collect::<Vec<_>>()
                    .into_iter()
                    .for_each(|i| {
                        self.process_encoded_audio(&i, false);
                    });
                self.voice_detections.clear();
                self.voice_detection_time = SystemTime::now();
                debug!("switching to feature detector");
                self.vad_enabled = false;
                self.process_encoded_audio(&resampled_audio, true)
            } else {
                if self.audio_cache.len() >= self.max_frames {
                    self.audio_cache
                        .drain(0..=self.audio_cache.len() - self.max_frames);
                }
                self.audio_cache.push(resampled_audio);
                None
            }
        } else {
            self.process_encoded_audio(&resampled_audio, true)
        }
    }
    fn process_encoded_audio(
        &mut self,
        audio_chunk: &[f32],
        run_detection: bool,
    ) -> Option<DetectedWakeword> {
        self.extractor.shift_and_filter_input(&audio_chunk);
        self.extractor.compute_frame_features();
        let features = self.extractor.features().to_vec();
        let mut detection: Option<DetectedWakeword> = None;
        self.frames.push(features);
        if self.frames.len() >= self.min_frames {
            if self.buffering {
                self.buffering = false;
                debug!("ready");
            }
            if run_detection {
                detection = self.run_detection();
            }
        }
        if self.frames.len() >= self.max_frames {
            self.frames.drain(0..=self.frames.len() - self.max_frames);
        }
        detection
    }
    fn extract_features_from_file(&mut self, file_path: String) -> Result<Vec<Vec<f32>>, String> {
        let path = Path::new(&file_path);
        if !path.exists() || !path.is_file() {
            warn!("File \"{}\" not found!", file_path);
            return Err("Can not read file".to_string());
        }

        let in_file = BufReader::new(File::open(file_path).or_else(|err| Err(err.to_string()))?);
        let wav_reader = WavReader::new(in_file).or_else(|err| Err(err.to_string()))?;
        let sample_rate = wav_reader.spec().sample_rate;
        let sample_format = wav_reader.spec().sample_format;
        let bits_per_sample = wav_reader.spec().bits_per_sample;
        let channels = wav_reader.spec().channels;
        if channels != 1 {
            return Err("Only samples with 1 channels are supported for now".to_string());
        }
        let audio_pcm_signed_resampled = match sample_format {
            SampleFormat::Int => {
                let bits_per_sample = bits_per_sample;
                assert!(bits_per_sample <= 32);
                let samples = wav_reader
                    .into_samples::<i32>()
                    .map(move |s| {
                        s.map(|s| {
                            if bits_per_sample < 16 {
                                (s << (16 - bits_per_sample)) as f32
                            } else {
                                (s >> (bits_per_sample - 16)) as f32
                            }
                        })
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                if sample_rate as usize != INTERNAL_SAMPLE_RATE {
                    resample_audio(sample_rate as usize, &samples)
                } else {
                    samples
                }
            }
            SampleFormat::Float => {
                let samples = wav_reader
                    .into_samples::<f32>()
                    .map(|s| s.map(|s| s * 32767.0).unwrap())
                    .collect::<Vec<_>>();
                if sample_rate as usize != INTERNAL_SAMPLE_RATE {
                    resample_audio(sample_rate as usize, &samples)
                } else {
                    samples
                }
            }
        };
        let mut is_silence = true;
        let mut feature_generator = DenoiseFeatures::new();
        let all_features = audio_pcm_signed_resampled
            .chunks_exact(nnnoiseless_fork::FRAME_SIZE)
            .into_iter()
            .filter_map(|audio_chuck| {
                feature_generator.shift_input(audio_chuck);
                let silence = feature_generator.compute_frame_features();
                if silence && is_silence {
                    None
                } else {
                    is_silence = false;
                    let features = feature_generator.features();
                    Some(features.to_vec())
                }
            })
            .collect::<Vec<_>>();
        Ok(self.normalize_features(all_features))
    }
    fn run_detection(&mut self) -> Option<DetectedWakeword> {
        let features = self.normalize_features(self.frames.to_vec());
        let result_option = self.get_best_keyword(features);
        match result_option {
            Some(result) => {
                if self.vad_enabled {
                    self.vad_enabled = false;
                    debug!("switching to feature detector");
                }
                self.voice_detection_time = SystemTime::now();
                if self.eager_mode {
                    if result.index != 0 {
                        debug!("Sorting '{}' templates", result.wakeword);
                        self.wakewords
                            .get_mut(&result.wakeword)
                            .unwrap()
                            .prioritize_template(result.index);
                    }
                    debug!(
                        "keyword '{}' detected, score {}",
                        result.wakeword, result.score
                    );
                    self.reset();
                    Some(result)
                } else {
                    if self.result_state.is_some() {
                        let prev_result = self.result_state.as_ref().unwrap();
                        let prev_wakeword = prev_result.wakeword.clone();
                        let prev_score = prev_result.score;
                        let prev_index = prev_result.index;
                        if result.wakeword == prev_wakeword && result.score < prev_score {
                            debug!(
                                "keyword '{}' detected, score {}",
                                result.wakeword, prev_score
                            );
                            self.reset();
                            return Some(DetectedWakeword {
                                wakeword: result.wakeword,
                                score: prev_score,
                                index: prev_index,
                            });
                        }
                    }
                    self.result_state = Some(result);
                    None
                }
            }
            None => {
                if self.result_state.is_some() {
                    let prev_result = self.result_state.as_ref().unwrap();
                    let wakeword = prev_result.wakeword.clone();
                    let score = prev_result.score;
                    let index = prev_result.index;
                    self.reset();
                    Some(DetectedWakeword {
                        wakeword,
                        score,
                        index,
                    })
                } else {
                    None
                }
            }
        }
    }
    fn reset(&mut self) {
        self.voice_detection_time = SystemTime::now();
        self.frames.clear();
        self.buffering = true;
        self.result_state = None;
    }
    fn normalize_features(&self, frames: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let num_frames = frames.len();
        if num_frames == 0 {
            return Vec::new();
        }
        let num_features = frames[0].len();
        let mut sum = Vec::with_capacity(num_features);
        sum.resize(num_features, 0.);
        let mut normalized_frames: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        normalized_frames.resize(num_frames, {
            let mut vector = Vec::with_capacity(num_features);
            vector.resize(num_features, 0.);
            vector
        });
        for i in 0..num_frames {
            for j in 0..num_features {
                sum[j] += frames[i][j];
                normalized_frames[i][j] = frames[i][j];
            }
        }
        for i in 0..num_frames {
            for j in 0..num_features {
                normalized_frames[i][j] = normalized_frames[i][j] - sum[j] / num_frames as f32
            }
        }
        normalized_frames
    }
    fn get_best_keyword(&self, features: Vec<Vec<f32>>) -> Option<DetectedWakeword> {
        let mut detections: Vec<DetectedWakeword> =
            if self.single_thread || self.wakewords.len() <= 1 {
                self.wakewords
                    .iter()
                    .filter_map(|(name, wakeword)| {
                        if !wakeword.is_enabled()
                            || self.result_state.is_some()
                                && self.result_state.as_ref().unwrap().wakeword != *name
                        {
                            return None;
                        }
                        let templates = wakeword.get_templates();
                        let averaged_template = wakeword.get_averaged_template();
                        let averaged_threshold = if self.result_state.is_none() {
                            wakeword
                                .get_averaged_threshold()
                                .unwrap_or(self.averaged_threshold)
                        } else {
                            0.
                        };
                        let threshold = wakeword.get_threshold().unwrap_or(self.threshold);
                        run_wakeword_detection(
                            &self.comparator,
                            &templates,
                            averaged_template,
                            self.eager_mode,
                            &features,
                            averaged_threshold,
                            threshold,
                            name.clone(),
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                self.wakewords
                    .iter()
                    .filter_map(|(name, wakeword)| {
                        if !wakeword.is_enabled()
                            || self.result_state.is_some()
                                && self.result_state.as_ref().unwrap().wakeword != *name
                        {
                            return None;
                        }
                        let wakeword_name = name.clone();
                        let threshold = wakeword.get_threshold().unwrap_or(self.threshold);
                        let averaged_threshold = if self.result_state.is_none() {
                            wakeword
                                .get_averaged_threshold()
                                .unwrap_or(self.averaged_threshold)
                        } else {
                            0.
                        };
                        let templates = wakeword.get_templates();
                        let averaged_template = wakeword.get_averaged_template();
                        let comparator = self.comparator.clone();
                        let eager_mode = self.eager_mode;
                        let features_copy = features.to_vec();
                        Some(thread::spawn(move || {
                            run_wakeword_detection(
                                &comparator,
                                &templates,
                                averaged_template,
                                eager_mode,
                                &features_copy,
                                averaged_threshold,
                                threshold,
                                wakeword_name,
                            )
                        }))
                    })
                    .map(JoinHandle::join)
                    .filter_map(Result::unwrap)
                    .collect::<Vec<_>>()
            };
        if detections.is_empty() {
            None
        } else {
            detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            Some(detections.remove(0))
        }
    }
}
fn resample_audio(input_sample_rate: usize, audio_pcm_signed: &[f32]) -> Vec<f32> {
    let mut resampler =
        FftFixedInOut::<f32>::new(input_sample_rate, INTERNAL_SAMPLE_RATE, 480, 1).unwrap();
    let mut out = resampler.output_buffer_allocate();
    let resampled_audio = audio_pcm_signed
        .chunks_exact(resampler.input_frames_next())
        .map(|sample| {
            resampler
                .process_into_buffer(&[sample], &mut out[..], None)
                .unwrap();
            out.get(0).unwrap().to_vec()
        })
        .flatten()
        .collect::<Vec<f32>>();
    resampled_audio
}

fn run_wakeword_detection(
    comparator: &FeatureComparator,
    templates: &[Vec<Vec<f32>>],
    averaged_template: Option<Vec<Vec<f32>>>,
    eager_mode: bool,
    features: &[Vec<f32>],
    averaged_threshold: f32,
    threshold: f32,
    wakeword_name: String,
) -> Option<DetectedWakeword> {
    if averaged_threshold > 0. && averaged_template.is_some() {
        let template = averaged_template.unwrap();
        let mut frames = features.to_vec();
        if frames.len() > template.len() {
            frames.drain(template.len()..frames.len());
        }
        let score = comparator.compare(
            &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
        );
        if score < averaged_threshold {
            return None;
        }
        debug!(
            "wakeword '{}' passes averaged detection with score {}",
            wakeword_name, score
        );
    }
    let mut detection: Option<DetectedWakeword> = None;
    for (index, template) in templates.iter().enumerate() {
        let mut frames = features.to_vec();
        if frames.len() > template.len() {
            frames.drain(template.len()..frames.len());
        }
        let score = comparator.compare(
            &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
        );
        debug!("wakeword '{}' scored {}", wakeword_name, score);
        if score < threshold {
            continue;
        }
        let prev_detection = detection.as_ref();
        if prev_detection.is_some() && score < prev_detection.unwrap().score {
            continue;
        }
        detection = Some(DetectedWakeword {
            wakeword: wakeword_name.clone(),
            score,
            index,
        });
        if eager_mode {
            break;
        }
    }
    detection
}

pub struct DetectedWakeword {
    // Detected wakeword name
    pub wakeword: String,
    // Detection score
    pub score: f32,
    // Detected wakeword template index
    pub index: usize,
}
impl Clone for DetectedWakeword {
    fn clone(&self) -> Self {
        Self {
            wakeword: self.wakeword.clone(),
            score: self.score.clone(),
            index: self.index.clone(),
        }
    }
}
