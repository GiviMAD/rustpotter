use crate::comparator::FeatureComparator;
use crate::wakeword::{Wakeword, WakewordModel, WakewordTemplate, WAKEWORD_MODEL_VERSION};
use hound::WavReader;
#[cfg(feature = "log")]
use log::{debug, warn};
use nnnoiseless::DenoiseFeatures;
use rubato::{FftFixedInOut, Resampler};
use savefile::{load_file, load_from_mem, save_file, save_to_mem};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
#[cfg(feature = "vad")]
use std::time::SystemTime;

pub(crate) const INTERNAL_SAMPLE_RATE: usize = 48000;
#[cfg(feature = "vad")]
/// Allowed vad modes
pub type VadMode = webrtc_vad::VadMode;
/// Allowed wav sample formats
pub type SampleFormat = hound::SampleFormat;

/// Dificulty for considering a frame as noise
#[derive(Clone)]
pub enum NoiseDetectionMode {
    Hardest,
    Hard,
    Normal,
    Easy,
    Easiest,
}
/// Supported endianness modes
#[derive(Clone)]
pub enum Endianness {
    Big,
    Little,
    Native,
}
/// This struct manages the wakeword generation and the spotting functionality.
///
/// ```
/// use rustpotter::{WakewordDetectorBuilder};
/// // assuming the audio input format match the detector defaults
/// let mut word_detector = WakewordDetectorBuilder::new().build();
/// // load and enable a wakeword
/// word_detector.add_wakeword_from_model_file("./tests/resources/oye_casa.rpw".to_owned(), true).unwrap();
/// let mut frame_buffer: Vec<i32> = vec![0; word_detector.get_samples_per_frame()];
/// // while true { Iterate forever
///     // fill the buffer with new samples...
///     let detection_option = word_detector.process(&frame_buffer);
///     if detection_option.is_some() {
///         let detection = detection_option.as_ref().unwrap();
///         println!(
///             "Detected '{}' with score {}!",
///             detection.wakeword,
///             detection.score,
///         );
///     }
/// // }
/// ```
pub struct WakewordDetector {
    // input options
    sample_format: SampleFormat,
    bits_per_sample: u16,
    channels: u16,
    endianness: Endianness,
    // detection options
    threshold: f32,
    averaged_threshold: f32,
    eager_mode: bool,
    single_thread: bool,
    resampler: Option<FftFixedInOut<f32>>,
    comparator: Arc<FeatureComparator>,
    #[cfg(feature = "vad")]
    vad_detector: Option<webrtc_vad::Vad>,
    #[cfg(feature = "vad")]
    vad_delay: u16,
    #[cfg(feature = "vad")]
    vad_sensitivity: f32,
    // state
    samples_per_frame: usize,
    bytes_per_frame: usize,
    buffering: bool,
    min_frames: usize,
    max_frames: usize,
    frames: Vec<Vec<f32>>,
    wakewords: HashMap<String, Wakeword>,
    result_state: Option<DetectedWakeword>,
    extractor: DenoiseFeatures,
    resampler_out_buffer: Option<Vec<Vec<f32>>>,
    noise_ref: f32,
    noise_sensitivity: f32,
    noise_detections: [u8; 2],
    #[cfg(feature = "vad")]
    vad_detection_enabled: bool,
    #[cfg(feature = "vad")]
    voice_detections: [u8; 2],
    #[cfg(feature = "vad")]
    voice_detection_time: SystemTime,
    #[cfg(feature = "vad")]
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
        channels: u16,
        endianness: Endianness,
        // detection options
        eager_mode: bool,
        single_thread: bool,
        threshold: f32,
        averaged_threshold: f32,
        comparator_band_size: usize,
        comparator_ref: f32,
        noise_mode: Option<NoiseDetectionMode>,
        noise_sensitivity: f32,
        #[cfg(feature = "vad")] vad_mode: Option<VadMode>,
        #[cfg(feature = "vad")] vad_delay: u16,
        #[cfg(feature = "vad")] vad_sensitivity: f32,
    ) -> Self {
        let mut samples_per_frame = 480 * channels as usize;
        let resampler = if sample_rate != INTERNAL_SAMPLE_RATE {
            let resampler =
                FftFixedInOut::<f32>::new(sample_rate, INTERNAL_SAMPLE_RATE, samples_per_frame, 1)
                    .unwrap();
            samples_per_frame = resampler.input_frames_next() * channels as usize;
            Some(resampler)
        } else {
            None
        };
        #[cfg(feature = "vad")]
        let vad_detector = vad_mode.map(|vad_mode| {
            webrtc_vad::Vad::new_with_rate_and_mode(webrtc_vad::SampleRate::Rate48kHz, vad_mode)
        });
        let detector = WakewordDetector {
            threshold,
            averaged_threshold,
            sample_format,
            bytes_per_frame: samples_per_frame * (bits_per_sample / 8) as usize,
            bits_per_sample,
            samples_per_frame,
            channels,
            endianness,
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
            noise_ref: noise_mode.map(get_noise_ref).unwrap_or(0.),
            noise_sensitivity,
            noise_detections: [0, 0],
            #[cfg(feature = "vad")]
            voice_detections: [0, 0],
            #[cfg(feature = "vad")]
            audio_cache: Vec::with_capacity(100),
            #[cfg(feature = "vad")]
            vad_detection_enabled: false,
            #[cfg(feature = "vad")]
            vad_detector,
            #[cfg(feature = "vad")]
            vad_delay,
            #[cfg(feature = "vad")]
            vad_sensitivity,
            #[cfg(feature = "vad")]
            voice_detection_time: SystemTime::UNIX_EPOCH,
        };
        detector
    }
    /// Loads a wakeword from its model bytes.
    pub fn add_wakeword_from_model_bytes(
        &mut self,
        bytes: Vec<u8>,
        enabled: bool,
    ) -> Result<String, Error> {
        let model: WakewordModel =
            load_from_mem(&bytes, WAKEWORD_MODEL_VERSION).or(Err(Error::new(ErrorKind::InvalidInput,"Invalid model bytes")))?;
        self.add_wakeword_from_model(model, enabled)
    }
    /// Loads a wakeword from its model path.
    pub fn add_wakeword_from_model_file(
        &mut self,
        path: String,
        enabled: bool,
    ) -> Result<String, Error> {
        let model: WakewordModel =
            load_file(&path, WAKEWORD_MODEL_VERSION).or(Err(Error::new(ErrorKind::InvalidInput,"Invalid model file: ".to_owned() + &path)))?;
        self.add_wakeword_from_model(model, enabled)
    }
    /// Generates the model file bytes from a loaded a wakeword.
    pub fn generate_wakeword_model_bytes(&self, name: String) -> Result<Vec<u8>, Error> {
        let model = self.get_wakeword_model(&name)?;
        save_to_mem(WAKEWORD_MODEL_VERSION, &model)
            .map_err(|_| Error::new(ErrorKind::InvalidInput,"Unable to generate model bytes"))
    }
    /// Generates a model file from a loaded a wakeword on the desired path.
    pub fn generate_wakeword_model_file(&self, name: String, path: String) -> Result<(), Error> {
        let model = self.get_wakeword_model(&name)?;
        save_file(path, WAKEWORD_MODEL_VERSION, &model)
            .map_err(|_| Error::new(ErrorKind::InvalidInput,"Unable to generate model file."))
    }
    /// Adds a wakeword using wav samples.
    ///
    /// ```
    /// use rustpotter::{WakewordDetectorBuilder};
    /// let mut word_detector = WakewordDetectorBuilder::new().build();
    /// word_detector.add_wakeword_with_wav_files(
    ///     "model_name",
    ///     true,
    ///     Some(0.35),
    ///     Some(0.6),
    ///     vec![],
    /// ).unwrap();
    /// // Save as model file
    /// // word_detector.generate_wakeword_model_file("model_name".to_owned(), "model_path".to_owned()).unwrap();
    /// ```
    pub fn add_wakeword_with_wav_files(
        &mut self,
        name: &str,
        enabled: bool,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
        samples: Vec<String>,
    ) -> Result<(), Error> {
        #[cfg(feature = "log")]
        debug!(
            "Adding wakeword \"{}\" (sample paths: {:?})",
            name, samples
        );
        if self.wakewords.get_mut(name).is_none() {
            self.wakewords.insert(
                name.to_string(),
                Wakeword::new(enabled, averaged_threshold, threshold),
            );
        }
        let mut errors: Vec<Result<(), Error>> = Vec::new();
        let samples_features = samples
            .iter()
            .map(|path| {
                let features_result = self.extract_features_from_wav_file(path);
                if let Ok(features) = features_result {
                    Ok((name.to_string(), features))
                }else {
                    Err(features_result.unwrap_err())
                }
            })
            .filter_map(|r| r.map_err(|e| errors.push(Err(e))).ok())
            .collect::<Vec<_>>();
            if !errors.is_empty() {
                let error: Result<(), Error> = errors.drain(0..=0).collect();
                return Err(error.unwrap_err());
            }
            self.add_wakeword_with_features(name, enabled, averaged_threshold, threshold, samples_features)
    }
    /// Adds a wakeword using wav samples.
    ///
    /// ```
    /// use rustpotter::{WakewordDetectorBuilder};
    /// let mut word_detector = WakewordDetectorBuilder::new().build();
    /// word_detector.add_wakeword_with_wav_buffers(
    ///     "model_name",
    ///     true,
    ///     Some(0.35),
    ///     Some(0.6),
    ///     vec![],
    /// );
    /// // Save as model file
    /// // word_detector.generate_wakeword_model_file("model_name".to_owned(), "model_path".to_owned()).unwrap();
    /// ```
    pub fn add_wakeword_with_wav_buffers(
        &mut self,
        name: &str,
        enabled: bool,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
        samples: Vec<(String, Vec<u8>)>,
    ) -> Result<(), Error> {
        #[cfg(feature = "log")]
        debug!(
            "Adding wakeword \"{}\"",
            name,
        );
        let mut errors: Vec<std::result::Result<_, Error>> = vec![];
        let samples_features = samples
            .iter()
            .map(|(name, buffer)| {
                let features_result = self.extract_features_from_wav_buffer(buffer.to_vec());
                if let Ok(features) = features_result {
                    Ok((name.to_string(), features))
                }else {
                    Err(features_result.unwrap_err())
                }
            })
            .filter_map(|r| r.map_err(|e| errors.push(Err(e))).ok())
            .collect::<Vec<_>>();
            if !errors.is_empty() {
                let error: Result<(), Error> = errors.drain(0..=0).collect();
                return Err(error.unwrap_err());
            }
            self.add_wakeword_with_features(name, enabled, averaged_threshold, threshold, samples_features)
    }
    fn add_wakeword_with_features(
        &mut self,
        name: &str,
        enabled: bool,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
        samples_features: Vec<(String, Vec<Vec<f32>>)>,
    ) -> Result<(), Error> {
        if self.wakewords.get_mut(name).is_none() {
            self.wakewords.insert(
                name.to_string(),
                Wakeword::new(enabled, averaged_threshold, threshold),
            );
        }
        let word = self.wakewords.get_mut(name).unwrap();
        word.add_templates_features(samples_features.to_vec());
        let templates = word.get_templates().to_vec();
        let averaged = if word.get_averaged_template().is_some() {
            Some(word.get_averaged_template().unwrap().to_vec())
        } else {
            None
        };
        self.update_detection_frame_size();
        templates.iter().for_each(|wakeword_template| {
            samples_features.iter().for_each(|(_name, _template)| {
                let _score = score_frame(
                    wakeword_template.get_template().to_vec(),
                    _template.to_vec(),
                    &self.comparator,
                );
                #[cfg(feature = "log")]
                debug!(
                    "Sample '{}' scored '{}' againts {}",
                    wakeword_template._get_name(),
                    _score,
                    _name
                );
            });
            if averaged.is_some() {
                let _score = score_frame(
                    wakeword_template.get_template().to_vec(),
                    averaged.as_ref().unwrap().to_vec(),
                    &self.comparator,
                );
                #[cfg(feature = "log")]
                debug!(
                    "Sample '{}' scored '{}' againts the averaged features",
                    wakeword_template._get_name(),
                    _score
                );
            }
        });
        Ok(())
    }
    /// Removes a wakeword by name.
    pub fn remove_wakeword(&mut self, name: &str) {
        self.wakewords.remove(name);
        self.update_detection_frame_size();
    }
    /// Sets detector threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
    /// Sets detector averaged threshold.
    pub fn set_averaged_threshold(&mut self, averaged_threshold: f32) {
        self.averaged_threshold = averaged_threshold;
    }
    /// Sets wakeword threshold.
    pub fn set_wakeword_threshold(&mut self, name: &str, threshold: f32) {
        let wakeword_option = self.wakewords.get_mut(name);
        if let Some(wakeword) = wakeword_option {
            wakeword.set_threshold(threshold);
        }
    }
    /// Sets wakeword averaged threshold.
    pub fn set_wakeword_averaged_threshold(&mut self, name: &str, averaged_threshold: f32) {
        let wakeword_option = self.wakewords.get_mut(name);
        if let Some(wakeword) = wakeword_option {
            wakeword.set_averaged_threshold(averaged_threshold);
        }
    }
    /// Process bytes buffer.
    ///
    /// Asserts that the buffer length should match the return
    /// of the get_bytes_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Assumes buffer endianness matches the configured for the detector.
    ///
    pub fn process_buffer(&mut self, audio_buffer: &[u8]) -> Option<DetectedWakeword> {
        assert!(audio_buffer.len() == self.get_bytes_per_frame());
        match self.sample_format {
            SampleFormat::Int => {
                let buffer_chunks = audio_buffer.chunks_exact((self.bits_per_sample / 8) as usize);
                let audio_chunk = match self.endianness {
                    Endianness::Little => buffer_chunks.map(|bytes| {
                        match self.bits_per_sample {
                            8 => {
                                i8::from_le_bytes([bytes[0]]) as i32
                            }
                            16 => {
                                i16::from_le_bytes([bytes[0], bytes[1]]) as i32
                            }
                            24 => {
                                i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]])
                            }
                            32 => {
                                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            }
                            _default => {
                                0
                            }
                        }}).collect::<Vec<i32>>(),
                        Endianness::Big => buffer_chunks.map(|bytes| {
                            match self.bits_per_sample {
                                8 => {
                                    i8::from_be_bytes([bytes[0]]) as i32
                                }
                                16 => {
                                    i16::from_be_bytes([bytes[0], bytes[1]]) as i32
                                }
                                24 => {
                                    i32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]])
                                }
                                32 => {
                                    i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                }
                                _default => {
                                    0
                                }
                            }}).collect::<Vec<i32>>(),
                        Endianness::Native =>  buffer_chunks.map(|bytes| {
                            match self.bits_per_sample {
                                8 => {
                                    i8::from_ne_bytes([bytes[0]]) as i32
                                }
                                16 => {
                                    i16::from_ne_bytes([bytes[0], bytes[1]]) as i32
                                }
                                24 => {
                                    i32::from_ne_bytes([0, bytes[0], bytes[1], bytes[2]])
                                }
                                32 => {
                                    i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                }
                                _default => {
                                    0
                                }
                            }}).collect::<Vec<i32>>(),
                };
                self.process_int(&audio_chunk)
            }
            SampleFormat::Float => {
                let buffer_chunks = audio_buffer.chunks_exact((self.bits_per_sample / 8) as usize);
                let audio_chunk: Vec<f32> = match self.endianness {
                    Endianness::Little => 
                    buffer_chunks
                    .map(|bytes| {
                        match self.bits_per_sample {
                            32 => {
                                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            }
                            _default => {
                                0.
                            }
                        }
                    }).collect::<Vec<f32>>(),
                    Endianness::Big => 
                    buffer_chunks
                    .map(|bytes| {
                        match self.bits_per_sample {
                            32 => {
                                f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            }
                            _default => {
                                0.
                            }
                        }
                    }).collect::<Vec<f32>>(),
                    Endianness::Native => 
                    buffer_chunks
                    .map(|bytes| {
                        match self.bits_per_sample {
                            32 => {
                                f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            }
                            _default => {
                                0.
                            }
                        }
                    }).collect::<Vec<f32>>(),
                };
                self.process_f32(&audio_chunk)
            }
        }
    }
    /// Process i32 audio chunks.
    ///
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Asserts that detector bits_per_sample is one of: 8, 16, 24, 32.
    ///
    /// Asserts that detector sample_format is 'int'.
    ///
    /// It's an alias for the process_i32 method.
    pub fn process(&mut self, audio_chunk: &[i32]) -> Option<DetectedWakeword> {
        self.process_i32(audio_chunk)
    }
    /// Process i8 audio chunks.
    ///
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Asserts that detector bits_per_sample is 8.
    ///
    /// Asserts that detector sample_format is 'int'.
    pub fn process_i8(&mut self, audio_chunk: &[i8]) -> Option<DetectedWakeword> {
        assert!(self.bits_per_sample == 8);
        self.process_int(&audio_chunk.iter().map(|i| *i as i32).collect::<Vec<i32>>())
    }
    /// Process i16 audio chunks.
    ///
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Asserts that detector bits_per_sample is one of: 8, 16.
    ///
    /// Asserts that detector sample_format is 'int'.
    pub fn process_i16(&mut self, audio_chunk: &[i16]) -> Option<DetectedWakeword> {
        assert!(self.bits_per_sample == 8 || self.bits_per_sample == 16);
        self.process_int(&audio_chunk.iter().map(|i| *i as i32).collect::<Vec<i32>>())
    }
    /// Process i32 audio chunks.
    ///
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Asserts that detector bits_per_sample is one of: 8, 16, 24, 32.
    ///
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
    ///
    /// Asserts that the audio chunk length should match the return
    /// of the get_samples_per_frame method.
    ///
    /// Assumes sample rate match the configured for the detector.
    ///
    /// Asserts that detector bits_per_sample is 32.
    ///
    /// Asserts that detector sample_format is 'float'.
    pub fn process_f32(&mut self, audio_chunk: &[f32]) -> Option<DetectedWakeword> {
        assert!(audio_chunk.len() == self.samples_per_frame);
        assert!(self.bits_per_sample == 32);
        assert!(self.sample_format == SampleFormat::Float);
        let float_buffer: Vec<f32> = if self.channels != 1 {
            audio_chunk
                .chunks_exact(self.channels as usize)
                .map(|chunk| chunk[0])
                .map(convert_f32_sample)
                .collect::<Vec<f32>>()
        } else {
            audio_chunk
                .iter()
                .map(convert_f32_sample_ref)
                .collect::<Vec<f32>>()
        };
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
        #[cfg(not(feature = "vad"))]
        return self.process_encoded_audio(&resampled_audio, true);
        #[cfg(feature = "vad")]
        return self.apply_vad_detection(resampled_audio);
    }
    /// Returns the desired chunk size.
    pub fn get_samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }
    /// Returns size in bytes for the desired chunk
    pub fn get_bytes_per_frame(&self) -> usize {
        self.bytes_per_frame
    }
    fn add_wakeword_from_model(
        &mut self,
        model: WakewordModel,
        enabled: bool,
    ) -> Result<String, Error> {
        let wakeword_name = String::from(model.get_name());
        let wakeword = Wakeword::from_model(model, enabled);
        self.wakewords.insert(wakeword_name.clone(), wakeword);
        self.update_detection_frame_size();
        Ok(wakeword_name)
    }

    fn update_detection_frame_size(&mut self) {
        let mut min_frames = 9999;
        let mut max_frames = 0;
        for (_, wakeword) in self.wakewords.iter() {
            if !wakeword.get_templates().is_empty() {
                min_frames = std::cmp::min(min_frames, wakeword.get_min_frames());
                max_frames = std::cmp::max(max_frames, wakeword.get_max_frames());
            }
        }
        #[cfg(feature = "log")]
        {
            debug!("{} min", min_frames);
            debug!("{} max", max_frames);
        }
        self.min_frames = min_frames;
        self.max_frames = max_frames;
    }
    fn get_wakeword_model(&self, name: &str) -> Result<WakewordModel, Error> {
        if let Some(wakeword) = self.wakewords.get(name) {
            Ok(WakewordModel::from_wakeword(name, wakeword))
        } else {
            Err(Error::new(ErrorKind::NotFound, "Missing wakeword"))
        }
    }
    fn process_int(&mut self, audio_chunk: &[i32]) -> Option<DetectedWakeword> {
        assert!(audio_chunk.len() == self.samples_per_frame);
        assert!(self.sample_format == SampleFormat::Int);
        let resampled_audio = if self.resampler.is_some() {
            let resampler = self.resampler.as_mut().unwrap();
            let bits_per_sample = self.bits_per_sample;
            let float_buffer = if self.channels != 1 {
                audio_chunk
                    .chunks_exact(self.channels as usize)
                    .map(|chunk| chunk[0])
                    .map(|s| {
                        if bits_per_sample < 16 {
                            (s << (16 - bits_per_sample)) as f32
                        } else {
                            (s >> (bits_per_sample - 16)) as f32
                        }
                    })
                    .collect::<Vec<f32>>()
            } else {
                audio_chunk
                    .iter()
                    .map(|s| {
                        if bits_per_sample < 16 {
                            (*s << (16 - bits_per_sample)) as f32
                        } else {
                            (*s >> (bits_per_sample - 16)) as f32
                        }
                    })
                    .collect::<Vec<f32>>()
            };
            let waves_in = vec![float_buffer; 1];
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            resampler
                .process_into_buffer(&waves_in, waves_out, None)
                .unwrap();
            let result = waves_out.get(0).unwrap();
            result.to_vec()
        } else {
            audio_chunk.iter().map(|n| *n as f32).collect::<Vec<f32>>()
        };
        #[cfg(not(feature = "vad"))]
        return self.process_encoded_audio(&resampled_audio, true);
        #[cfg(feature = "vad")]
        return self.apply_vad_detection(resampled_audio);
    }
    #[cfg(feature = "vad")]
    fn apply_vad_detection(&mut self, resampled_audio: Vec<f32>) -> Option<DetectedWakeword> {
        if self.result_state.is_none()
            && self.vad_detector.is_some()
            && (self.vad_detection_enabled
                || self.voice_detection_time.elapsed().unwrap().as_secs() >= self.vad_delay as u64)
        {
            let vad = self.vad_detector.as_mut().unwrap();
            if !self.vad_detection_enabled {
                #[cfg(feature = "log")]
                debug!("switching to vad detector");
                self.vad_detection_enabled = true;
            }
            let is_voice_result = vad.is_voice_segment(
                &resampled_audio
                    .iter()
                    .map(|i| *i as i16)
                    .collect::<Vec<i16>>(),
            );
            let is_voice_frame = is_voice_result.is_err() || is_voice_result.unwrap();
            if is_voice_frame {
                if self.voice_detections[0] != 100 {
                    self.voice_detections[0] += 1;
                    if self.voice_detections[0] + self.voice_detections[1] > 100 {
                        self.voice_detections[1] -= 1;
                    }
                }
            } else {
                if self.voice_detections[1] != 100 {
                    self.voice_detections[1] += 1;
                    if self.voice_detections[0] + self.voice_detections[1] > 100 {
                        self.voice_detections[0] -= 1;
                    }
                }
            }
            if self.voice_detections[0] + self.voice_detections[1] < 50 {
                return self.process_encoded_audio(&resampled_audio, true);
            }
            if self.voice_detections[0]
                >= (self.vad_sensitivity
                    * (self.voice_detections[0] + self.voice_detections[1]) as f32)
                    as u8
            {
                #[cfg(feature = "log")]
                debug!("voice detected; processing cache");
                self.audio_cache
                    .drain(0..self.audio_cache.len())
                    .collect::<Vec<Vec<f32>>>()
                    .into_iter()
                    .for_each(|i| {
                        self.process_encoded_audio(&i, false);
                    });
                self.voice_detections[0] = 0;
                self.voice_detections[1] = 0;
                self.voice_detection_time = SystemTime::now();
                #[cfg(feature = "log")]
                debug!("switching to feature detector");
                self.vad_detection_enabled = false;
                self.process_encoded_audio(&resampled_audio, true)
            } else {
                if self.max_frames != 0 {
                    if self.audio_cache.len() >= self.max_frames {
                        self.audio_cache
                            .drain(0..=self.audio_cache.len() - self.max_frames);
                    }
                    self.audio_cache.push(resampled_audio);
                }
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
        self.extractor.shift_and_filter_input(audio_chunk);
        self.extractor.compute_frame_features();
        let noise_level: f32 = self.extractor.ex.iter().sum();
        let silence_detected = if self.noise_ref != 0. && self.result_state.is_none() {
            let is_noise_frame = noise_level > self.noise_ref;
            if is_noise_frame {
                if self.noise_detections[0] != 100 {
                    self.noise_detections[0] += 1;
                    if self.noise_detections[0] + self.noise_detections[1] > 100 {
                        self.noise_detections[1] -= 1;
                    }
                }
            } else if self.noise_detections[1] != 100 {
                self.noise_detections[1] += 1;
                if self.noise_detections[0] + self.noise_detections[1] > 100 {
                    self.noise_detections[0] -= 1;
                }
            }
            if self.noise_detections[0] + self.noise_detections[1] < 100 {
                false
            } else {
                let noise_detected = self.noise_detections[0]
                    >= (self.noise_sensitivity
                        * (self.noise_detections[0] + self.noise_detections[1]) as f32)
                        as u8;
                if noise_detected {
                    #[cfg(feature = "log")]
                    debug!("noise detected");
                    self.noise_detections[0] = 0;
                    self.noise_detections[1] = 0;
                }
                !noise_detected
            }
        } else {
            false
        };
        let features = self.extractor.features().to_vec();
        let mut detection: Option<DetectedWakeword> = None;
        if !self.wakewords.is_empty() && self.max_frames != 0 {
            self.frames.push(features);
            if self.frames.len() >= self.min_frames {
                if self.buffering {
                    self.buffering = false;
                    #[cfg(feature = "log")]
                    debug!("ready");
                }
                if run_detection && !silence_detected {
                    detection = self.run_detection();
                }
            }
            if self.frames.len() >= self.max_frames {
                self.frames.drain(0..=self.frames.len() - self.max_frames);
            }
        }
        detection
    }
    fn extract_features_from_wav_buffer(&mut self, buffer: Vec<u8>) -> Result<Vec<Vec<f32>>, Error> {
        let buf_reader = BufReader::new(buffer.as_slice());
        self.extract_features_from_wav_buffer_reader(buf_reader)
    }
    fn extract_features_from_wav_file(&mut self, file_path: &str) -> Result<Vec<Vec<f32>>, Error> {
        let path = Path::new(file_path);
        if !path.exists() || !path.is_file() {
            #[cfg(feature = "log")]
            warn!("File \"{}\" not found!", file_path);
            return Err(Error::new(ErrorKind::NotFound, "File not found: ".to_owned() + file_path))
        }
        let file = File::open(file_path).map_err(|_| Error::new(ErrorKind::NotFound, "Can not open file: ".to_owned() + file_path))?;
        let buf_reader = BufReader::new(file);
        self.extract_features_from_wav_buffer_reader(buf_reader)
    }
    fn extract_features_from_wav_buffer_reader<R: std::io::Read>(&mut self, buf_reader: BufReader<R>) -> Result<Vec<Vec<f32>>, Error> {
        let wav_reader = WavReader::new(buf_reader).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid wav data provided."))?;
        let sample_rate = wav_reader.spec().sample_rate;
        let sample_format = wav_reader.spec().sample_format;
        let bits_per_sample = wav_reader.spec().bits_per_sample;
        let channels = wav_reader.spec().channels;
        let audio_pcm_signed_resampled = match sample_format {
            SampleFormat::Int => {
                let bits_per_sample = bits_per_sample;
                assert!(bits_per_sample <= 32);
                let samples = wav_reader
                    .into_samples::<i32>()
                    .collect::<Vec<_>>()
                    .chunks_exact(channels as usize)
                    .map(|chunk| chunk[0].as_ref().unwrap())
                    .map(|s| {
                        if bits_per_sample < 16 {
                            (*s << (16 - bits_per_sample)) as f32
                        } else {
                            (*s >> (bits_per_sample - 16)) as f32
                        }
                    })
                    .collect::<Vec<f32>>();
                if sample_rate as usize != INTERNAL_SAMPLE_RATE {
                    resample_audio(sample_rate as usize, &samples)
                } else {
                    samples
                }
            }
            SampleFormat::Float => {
                let samples = wav_reader
                    .into_samples::<f32>()
                    .collect::<Vec<_>>()
                    .chunks_exact(channels as usize)
                    .map(|chunk| chunk[0].as_ref().unwrap())
                    .map(convert_f32_sample_ref)
                    .collect::<Vec<f32>>();
                if sample_rate as usize != INTERNAL_SAMPLE_RATE {
                    resample_audio(sample_rate as usize, &samples)
                } else {
                    samples
                }
            }
        };
        let mut is_silence = true;
        let mut feature_extractor = DenoiseFeatures::new();
        let all_features = audio_pcm_signed_resampled
            .chunks_exact(nnnoiseless::FRAME_SIZE)
            .into_iter()
            .filter_map(|audio_chuck| {
                feature_extractor.shift_input(audio_chuck);
                feature_extractor.compute_frame_features();
                let noise_level: f32 = feature_extractor.ex.iter().sum();
                if noise_level < 0.04 && is_silence {
                    None
                } else {
                    is_silence = false;
                    let features = feature_extractor.features();
                    Some(features.to_vec())
                }
            })
            .collect::<Vec<Vec<f32>>>();
        Ok(self.normalize_features(all_features))
    }
    fn run_detection(&mut self) -> Option<DetectedWakeword> {
        let features = self.normalize_features(self.frames.to_vec());
        let result_option = self.get_best_wakeword(features);
        match result_option {
            Some(result) => {
                #[cfg(feature = "vad")]
                {
                    if self.vad_detection_enabled {
                        self.vad_detection_enabled = false;
                        #[cfg(feature = "log")]
                        debug!("switching to feature detector");
                    }
                    self.voice_detections[0] = 0;
                    self.voice_detections[1] = 0;
                    self.voice_detection_time = SystemTime::now();
                }
                if self.eager_mode {
                    if result.index != 0 {
                        #[cfg(feature = "log")]
                        debug!("Sorting '{}' templates", result.wakeword);
                        self.wakewords
                            .get_mut(&result.wakeword)
                            .unwrap()
                            .prioritize_template(result.index);
                    }
                    #[cfg(feature = "log")]
                    debug!(
                        "wakeword '{}' detected, score {}",
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
                            #[cfg(feature = "log")]
                            debug!(
                                "wakeword '{}' detected, score {}",
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
        #[cfg(feature = "vad")]
        {
            self.voice_detections[0] = 0;
            self.voice_detections[1] = 0;
            self.voice_detection_time = SystemTime::now();
        }
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
        for (i, normalized_frames_item) in normalized_frames.iter_mut().enumerate().take(num_frames)
        {
            for (j, sum_item) in sum.iter_mut().enumerate().take(num_features) {
                let value = frames[i][j];
                *sum_item += value;
                normalized_frames_item[j] = value;
            }
        }
        for normalized_frames_item in normalized_frames.iter_mut().take(num_frames) {
            for (j, sum_item) in sum.iter().enumerate().take(num_features) {
                normalized_frames_item[j] -= sum_item / num_frames as f32
            }
        }
        normalized_frames
    }
    fn get_best_wakeword(&self, features: Vec<Vec<f32>>) -> Option<DetectedWakeword> {
        let mut detections: Vec<DetectedWakeword> =
            if self.single_thread || self.wakewords.len() <= 1 {
                self.wakewords
                    .iter()
                    .filter_map(|(name, wakeword)| {
                        if !wakeword.is_enabled() {
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
                    .collect::<Vec<DetectedWakeword>>()
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
                    .collect::<Vec<DetectedWakeword>>()
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
        .flat_map(|sample| {
            resampler
                .process_into_buffer(&[sample], &mut out[..], None)
                .unwrap();
            out.get(0).unwrap().to_vec()
        })
        .collect::<Vec<f32>>();
    resampled_audio
}

fn run_wakeword_detection(
    comparator: &FeatureComparator,
    templates: &[WakewordTemplate],
    averaged_template: Option<Vec<Vec<f32>>>,
    eager_mode: bool,
    features: &[Vec<f32>],
    averaged_threshold: f32,
    threshold: f32,
    wakeword_name: String,
) -> Option<DetectedWakeword> {
    if averaged_threshold > 0. {
        if let Some(template) = averaged_template {
            let score = score_frame(features.to_vec(), template, comparator);
            if score < averaged_threshold {
                return None;
            }
            #[cfg(feature = "log")]
            debug!(
                "wakeword '{}' passes averaged detection with score {}",
                wakeword_name, score
            );
        }
    }
    let mut detection: Option<DetectedWakeword> = None;
    for (index, template) in templates.iter().enumerate() {
        let score = score_frame(
            features.to_vec(),
            template.get_template().to_vec(),
            comparator,
        );
        #[cfg(feature = "log")]
        debug!(
            "wakeword '{}' scored {} - template '{}'",
            wakeword_name,
            score,
            template._get_name()
        );
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
#[inline(always)]
fn score_frame(
    mut frame_features: Vec<Vec<f32>>,
    template: Vec<Vec<f32>>,
    comparator: &FeatureComparator,
) -> f32 {
    if frame_features.len() > template.len() {
        frame_features.drain(template.len()..frame_features.len());
    }
    let score = comparator.compare(
        &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
        &frame_features
            .iter()
            .map(|item| &item[..])
            .collect::<Vec<_>>(),
    );
    score
}
fn get_noise_ref(mode: NoiseDetectionMode) -> f32 {
    match mode {
        NoiseDetectionMode::Hardest => 45000.,
        NoiseDetectionMode::Hard => 30000.,
        NoiseDetectionMode::Normal => 15000.,
        NoiseDetectionMode::Easy => 9000.,
        NoiseDetectionMode::Easiest => 3000.,
    }
}
#[inline(always)]
fn convert_f32_sample_ref(value: &f32) -> f32 {
    convert_f32_sample(*value)
}
#[inline(always)]
fn convert_f32_sample(value: f32) -> f32 {
    let ranged_value = if value < -1. {
        -1.
    } else if value > 1. {
        1.
    } else {
        value
    };
    ranged_value * 32767.5 - 0.5
}
/// Represents a successful wakeword detection.
pub struct DetectedWakeword {
    /// Detected wakeword name.
    pub wakeword: String,
    /// Detection score.
    pub score: f32,
    /// Detected wakeword template index.
    pub index: usize,
}
impl Clone for DetectedWakeword {
    fn clone(&self) -> Self {
        Self {
            wakeword: self.wakeword.clone(),
            score: self.score,
            index: self.index,
        }
    }
}
