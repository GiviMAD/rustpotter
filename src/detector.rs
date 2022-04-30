use crate::comparator::FeatureComparator;
use crate::nnnoiseless_fork::DenoiseFeatures;
use crate::wakeword::{Wakeword, WakewordModel};
use log::{debug, info, warn};
use rubato::{FftFixedInOut, Resampler};
use savefile::{load_file, save_file, save_to_mem};
use std::path::Path;
use std::thread;
use std::{collections::HashMap, fs};

static INTERNAL_SAMPLE_RATE: usize = 48000;
pub struct FeatureDetectorBuilder {
    threshold: Option<f32>,
    sample_rate: Option<usize>,
    comparator_band_size: Option<usize>,
    comparator_ref: Option<f32>,
}
impl FeatureDetectorBuilder {
    pub fn new() -> Self {
        FeatureDetectorBuilder {
            threshold: None,
            sample_rate: None,
            comparator_band_size: None,
            comparator_ref: None,
        }
    }
    pub fn build(&self) -> FeatureDetector {
        FeatureDetector::new(
            self.get_threshold(),
            self.get_sample_rate(),
            self.get_comparator_band_size(),
            self.get_comparator_ref(),
        )
    }
    pub fn get_threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5)
    }
    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate.unwrap_or(INTERNAL_SAMPLE_RATE)
    }
    pub fn get_comparator_band_size(&self) -> usize {
        self.comparator_band_size.unwrap_or(10)
    }
    pub fn get_comparator_ref(&self) -> f32 {
        self.comparator_ref.unwrap_or(0.22)
    }
    pub fn set_threshold(&mut self, value: f32) {
        self.threshold = Some(value);
    }
    pub fn set_sample_rate(&mut self, value: usize) {
        self.sample_rate = Some(value);
    }
    pub fn set_comparator_band_size(&mut self, value: usize) {
        self.comparator_band_size = Some(value);
    }
    pub fn set_comparator_ref(&mut self, value: f32) {
        self.comparator_ref = Some(value);
    }
}
pub struct FeatureDetector {
    // options
    threshold: f32,
    comparator_band_size: usize,
    comparator_ref: f32,
    sample_rate: usize,
    // state
    samples_per_frame: usize,
    buffering: bool,
    min_frames: usize,
    max_frames: usize,
    frames: Vec<Vec<f32>>,
    keywords: HashMap<String, Wakeword>,
    result_state: Option<DetectedWakeword>,
    extractor: DenoiseFeatures,
    resampler: Option<FftFixedInOut<f32>>,
    resampler_out_buffer: Option<Vec<Vec<f32>>>,
}
impl FeatureDetector {
    pub fn new(
        threshold: f32,
        sample_rate: usize,
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
        let detector = FeatureDetector {
            threshold,
            sample_rate,
            samples_per_frame,
            frames: Vec::new(),
            keywords: HashMap::new(),
            buffering: true,
            min_frames: 9999,
            max_frames: 0,
            result_state: None,
            extractor: DenoiseFeatures::new(),
            comparator_band_size,
            comparator_ref,
            resampler_out_buffer: if resampler.is_some() {
                Some(resampler.as_ref().unwrap().output_buffer_allocate())
            } else {
                None
            },
            resampler,
        };
        detector
    }
    pub fn add_keyword_from_model(
        &mut self,
        path: String,
        enable_average: bool,
        enabled: bool,
    ) -> Result<(), String> {
        let model: WakewordModel = load_file(path, 0).or(Err("Unable to load model data"))?;
        let keyword = model.keyword.clone();
        let wakeword = Wakeword::from_model(model, enable_average, enabled);
        self.update_detection_frame_size(wakeword.get_min_frames(), wakeword.get_max_frames());
        self.keywords.insert(keyword, wakeword);
        Ok(())
    }
    fn update_detection_frame_size(&mut self, min_frames: usize, max_frames: usize) {
        self.min_frames = std::cmp::min(self.min_frames, min_frames);
        self.max_frames = std::cmp::max(self.max_frames, max_frames);
    }
    pub fn add_keyword(
        &mut self,
        keyword: String,
        enable_average: bool,
        enabled: bool,
        threshold: Option<f32>,
        templates: Vec<String>,
    ) {
        info!(
            "Adding keyword \"{}\" (templates: {:?})",
            keyword, templates
        );
        if self.keywords.get_mut(&keyword).is_none() {
            self.keywords.insert(
                keyword.clone(),
                Wakeword::new(enable_average, enabled, threshold),
            );
        }
        let mut min_frames: usize = 0;
        let mut max_frames: usize = 0;
        for template in templates {
            match self.extract_features_from_file(template) {
                Ok(features) => {
                    let word = self.keywords.get_mut(&keyword).unwrap();
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
    pub fn create_wakeword_model(&self, name: String, path: String) -> Result<(), String> {
        let model = self.get_wakeword_model(&name)?;
        save_file(path, 0, &model).or(Err(String::from("Unable to generate file")))
    }
    pub fn _create_wakeword_model_bytes(&self, name: String) -> Result<Vec<u8>, String> {
        let model = self.get_wakeword_model(&name)?;
        save_to_mem(0, &model).or(Err(String::from("Unable to generate model bytes")))
    }
    fn get_wakeword_model(&self, name: &String) -> Result<WakewordModel, String> {
        let keyword = self.keywords.get(name);
        if keyword.is_none() {
            Err(String::from("Missing wakeword"))
        } else {
            let features = keyword.unwrap().get_templates();
            let model = WakewordModel::new(
                name.clone(),
                features,
                keyword.unwrap().get_threshold(),
            );
            Ok(model)
        }
    }
    pub fn process_pcm_signed(&mut self, audio_chunk: &[i16]) -> Option<DetectedWakeword> {
        if audio_chunk.len() != self.samples_per_frame {
            panic!("Invalid input length {} ", audio_chunk.len());
        }
        let resampled_audio = if self.resampler.is_some() {
            let resampler = self.resampler.as_mut().unwrap();
            let mut float_buffer: Vec<f32> = Vec::with_capacity(audio_chunk.len());
            for value in audio_chunk {
                float_buffer.push(*value as f32);
            }
            let waves_in = vec![float_buffer; 1];
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            resampler.process_into_buffer(&waves_in, waves_out,None).unwrap();
            let result = waves_out.get(0).unwrap();
            result.to_vec()
        } else {
            audio_chunk.into_iter().map(|n| *n as f32).collect::<Vec<_>>()
        };
        self.extractor.shift_and_filter_input(&resampled_audio);
        self.extractor.compute_frame_features();
        // TODO: use silence to avoid further computation
        let features = self.extractor.features().to_vec();
        self.process_new_feature_vec(features)
    }
    fn extract_features_from_file(&mut self, file_path: String) -> Result<Vec<Vec<f32>>, &str> {
        let path = Path::new(&file_path);
        if !path.exists() || !path.is_file() {
            warn!("File \"{}\" not found!", file_path);
            return Err("Can not read file");
        }
        match fs::read(path) {
            Ok(input) => {
                let mut audio_bytes = input.to_vec();
                audio_bytes.drain(0..44);
                let mut feature_generator  = DenoiseFeatures::new();
                let audio_pcm_signed = audio_bytes
                .chunks_exact(2)
                .into_iter()
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32)
                .collect::<Vec<_>>();
                let audio_pcm_signed_resampled = if self.sample_rate == INTERNAL_SAMPLE_RATE {
                    audio_pcm_signed
                } else {
                    let mut resampler = FftFixedInOut::<f32>::new(self.sample_rate, INTERNAL_SAMPLE_RATE, 480, 1)
                    .unwrap();
                    let mut out = resampler.output_buffer_allocate();
                   let resampled_audio = audio_pcm_signed.chunks_exact(resampler.input_frames_next()).map(|sample|{
                        resampler.process_into_buffer(&[sample],&mut out[..] ,None).unwrap();
                        out.get(0).unwrap().to_vec()
                    }).flatten().collect::<Vec<f32>>();
                    resampled_audio
                };
                let mut is_silence = true;
                let all_features = audio_pcm_signed_resampled.chunks_exact(nnnoiseless::FRAME_SIZE).into_iter()
                .filter_map(|audio_chuck|{
                    feature_generator.shift_input(audio_chuck);
                    let silence = feature_generator.compute_frame_features();
                    if silence && is_silence {
                        None
                    } else {
                        is_silence = false;
                        let features = feature_generator.features();
                        Some(features.to_vec())
                    }
                }).collect::<Vec<_>>();
                Ok(self.normalize_features(all_features))
            }
            Err(..) => {
                warn!("Can not read file file \"{}\"", file_path);
                Err("Can not read file file")
            }
        }
    }
    fn process_new_feature_vec(&mut self, features: Vec<f32>) -> Option<DetectedWakeword> {
        let mut result: Option<DetectedWakeword> = None;
        self.frames.push(features);
        if self.frames.len() >= self.min_frames {
            if self.buffering {
                self.buffering = false;
                println!("Ready");
            }
            result = self.run_detection();
        }
        if self.frames.len() >= self.max_frames {
            self.frames.drain(0..1);
        }
        result
    }
    fn run_detection(&mut self) -> Option<DetectedWakeword> {
        let features = self.normalize_features(self.frames.to_vec());
        let result_option = self.get_best_keyword(features);
        match result_option {
            Some(result) => {
                if self.result_state.is_some() {
                    let prev_wakeword = self.result_state.as_ref().unwrap().wakeword.clone();
                    let prev_score = self.result_state.as_ref().unwrap().score;
                    if result.wakeword == prev_wakeword && result.score < prev_score {
                        debug!(
                            "keyword '{}' detected, score {}",
                            result.wakeword, prev_score
                        );
                        self.reset();
                        return Some(DetectedWakeword {
                            wakeword: result.wakeword,
                            score: prev_score,
                        });
                    }
                }
                self.result_state = Some(result);
                None
            }
            None => None,
        }
    }
    fn reset(&mut self) {
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
        let mut results: Vec<thread::JoinHandle<Option<DetectedWakeword>>> = vec![];
        for (name, keyword) in &self.keywords {
            if !keyword.is_enabled() {
                continue;
            }
            let wakeword_name = name.clone();
            let threshold = keyword.get_threshold().unwrap_or(self.threshold);
            let templates = keyword.get_templates();
            let features_copy = features.to_vec();
            let comparator_band_size = self.comparator_band_size;
            let comparator_ref = self.comparator_ref;
            let comparator_task = thread::spawn(move || {
                let comparator = FeatureComparator::new(comparator_band_size, comparator_ref);
                let mut detection: Option<DetectedWakeword> = None;
                for template in templates {
                    let mut frames = features_copy.to_vec();
                    if frames.len() > template.len() {
                        frames.drain(template.len()..frames.len());
                    }
                    let score = comparator.compare(
                        &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
                        &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
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
                    });
                }
                detection
            });
            results.push(comparator_task);
        }
        let mut detections: Vec<DetectedWakeword> = vec![];
        for result_task in results {
            let result = result_task.join().unwrap();
            if result.is_some() {
                detections.push(result.unwrap());
            }
        }
        if detections.is_empty() {
            None
        } else {
            detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            Some(detections.remove(0))
        }
    }
    pub fn get_samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }
}

pub struct DetectedWakeword {
    pub wakeword: String,
    pub score: f32,
}
impl Clone for DetectedWakeword {
    fn clone(&self) -> Self {
        Self {
            wakeword: self.wakeword.clone(),
            score: self.score.clone(),
        }
    }
}
