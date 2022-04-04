use crate::comparator::FeatureComparator;
use crate::extractor::FeatureExtractor;
use crate::wakeword::{Wakeword, WakewordModel};
use log::{debug, info, warn};
use savefile::{load_file, save_file, save_to_mem};
use std::path::Path;
use std::thread;
use std::{collections::HashMap, fs};
pub struct FeatureDetectorBuilder {
    threshold: Option<f32>,
    sample_rate: Option<usize>,
    bit_length: Option<usize>,
    frame_length_ms: Option<usize>,
    frame_shift_ms: Option<usize>,
    num_coefficients: Option<usize>,
    pre_emphasis_coefficient: Option<f32>,
    comparator_band_size: Option<usize>,
    comparator_ref: Option<f32>,
}
impl FeatureDetectorBuilder {
    pub fn new() -> Self {
        FeatureDetectorBuilder {
            threshold: None,
            sample_rate: None,
            bit_length: None,
            frame_length_ms: None,
            frame_shift_ms: None,
            num_coefficients: None,
            pre_emphasis_coefficient: None,
            comparator_band_size: None,
            comparator_ref: None,
        }
    }
    pub fn build(&self) -> FeatureDetector {
        FeatureDetector::new(
            self.get_threshold(),
            self.get_sample_rate(),
            self.get_bit_length(),
            self.get_samples_per_frame(),
            self.get_samples_per_shift(),
            self.get_num_coefficients(),
            self.get_pre_emphasis_coefficient(),
            self.get_comparator_band_size(),
            self.get_comparator_ref(),
        )
    }
    pub fn get_threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5)
    }
    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate.unwrap_or(16000)
    }
    pub fn get_bit_length(&self) -> usize {
        self.bit_length.unwrap_or(16)
    }
    pub fn get_frame_length_ms(&self) -> usize {
        self.frame_length_ms.unwrap_or(30)
    }
    pub fn get_frame_shift_ms(&self) -> usize {
        self.frame_shift_ms.unwrap_or(10)
    }
    pub fn get_num_coefficients(&self) -> usize {
        self.num_coefficients.unwrap_or(13)
    }
    pub fn get_pre_emphasis_coefficient(&self) -> f32 {
        self.pre_emphasis_coefficient.unwrap_or(0.97)
    }
    pub fn get_comparator_band_size(&self) -> usize {
        self.comparator_band_size.unwrap_or(5)
    }
    pub fn get_comparator_ref(&self) -> f32 {
        self.comparator_ref.unwrap_or(0.22)
    }
    pub fn get_samples_per_frame(&self) -> usize {
        self.get_sample_rate() * self.get_frame_length_ms() / 1000
    }
    pub fn get_samples_per_shift(&self) -> usize {
        self.get_sample_rate() * self.get_frame_shift_ms() / 1000
    }
    pub fn set_threshold(&mut self, value: f32) {
        self.threshold = Some(value);
    }
    pub fn set_sample_rate(&mut self, value: usize) {
        self.sample_rate = Some(value);
    }
    pub fn set_bit_length(&mut self, value: usize) {
        self.bit_length = Some(value);
    }
    pub fn set_frame_length_ms(&mut self, value: usize) {
        self.frame_length_ms = Some(value);
    }
    pub fn set_frame_shift_ms(&mut self, value: usize) {
        self.frame_shift_ms = Some(value);
    }
    pub fn set_num_coefficients(&mut self, value: usize) {
        self.num_coefficients = Some(value);
    }
    pub fn set_pre_emphasis_coefficient(&mut self, value: f32) {
        self.pre_emphasis_coefficient = Some(value);
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
    sample_rate: usize,
    bit_length: usize,
    samples_per_frame: usize,
    samples_per_shift: usize,
    num_coefficients: usize,
    pre_emphasis_coefficient: f32,
    comparator_band_size: usize,
    comparator_ref: f32,
    // state
    buffering: bool,
    min_frames: usize,
    max_frames: usize,
    frames: Vec<Vec<f32>>,
    keywords: HashMap<String, Wakeword>,
    result_state: Option<DetectedWakeword>,
    extractor: FeatureExtractor,
}
impl FeatureDetector {
    pub fn new(
        threshold: f32,
        sample_rate: usize,
        bit_length: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: usize,
        pre_emphasis_coefficient: f32,
        comparator_band_size: usize,
        comparator_ref: f32,
    ) -> FeatureDetector {
        let detector = FeatureDetector {
            threshold,
            sample_rate,
            bit_length,
            samples_per_frame,
            samples_per_shift,
            num_coefficients,
            pre_emphasis_coefficient,
            frames: Vec::new(),
            keywords: HashMap::new(),
            buffering: true,
            min_frames: 9999,
            max_frames: 0,
            result_state: None,
            extractor: FeatureExtractor::new(
                sample_rate,
                samples_per_frame,
                samples_per_shift,
                num_coefficients,
                pre_emphasis_coefficient,
            ),
            comparator_band_size,
            comparator_ref,
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
        if model.sample_rate != self.sample_rate {
            return Err(format!(
                "Invalid model: sample_rate is {}",
                model.sample_rate
            ));
        }
        if model.bit_length != self.bit_length {
            return Err(format!("Invalid model: bit_length is {}", model.bit_length));
        }
        if model.channels != 1 {
            return Err(format!("Invalid model: channels is {}", model.channels));
        }
        if model.samples_per_frame != self.samples_per_frame {
            return Err(format!(
                "Invalid model: samples_per_frame is {}",
                model.samples_per_frame
            ));
        }
        if model.samples_per_shift != self.samples_per_shift {
            return Err(format!(
                "Invalid model: samples_per_shift is {}",
                model.samples_per_shift
            ));
        }
        if model.num_coefficients != self.num_coefficients {
            return Err(format!(
                "Invalid model: num_coefficients is {}",
                model.num_coefficients
            ));
        }
        if model.pre_emphasis_coefficient != self.pre_emphasis_coefficient {
            return Err(format!(
                "Invalid model: pre_emphasis_coefficient is {}",
                model.pre_emphasis_coefficient
            ));
        }
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
                self.sample_rate,
                self.bit_length,
                1,
                self.samples_per_frame,
                self.samples_per_shift,
                self.num_coefficients,
                self.pre_emphasis_coefficient,
                keyword.unwrap().get_threshold(),
            );
            Ok(model)
        }
    }
    pub fn process_bytes(&mut self, buffer: Vec<u8>) -> Vec<DetectedWakeword> {
        let features = self.extractor.process_buffer(&buffer);
        self.process_features(features)
    }
    pub fn process_pcm_signed(&mut self, buffer: Vec<i16>) -> Vec<DetectedWakeword> {
        let features = self.extractor.process_audio(&buffer);
        self.process_features(features)
    }
    fn process_features(&mut self, features: Vec<Vec<f32>>) -> Vec<DetectedWakeword> {
        features
            .into_iter()
            .map(|feature| self.process_new_feature_vec(feature))
            .filter(Option::is_some)
            .map(Option::unwrap)
            .collect::<Vec<DetectedWakeword>>()
    }
    fn extract_features_from_file(&mut self, file_path: String) -> Result<Vec<Vec<f32>>, &str> {
        let path = Path::new(&file_path);
        if !path.exists() || !path.is_file() {
            warn!("File \"{}\" not found!", file_path);
            return Err("Can not read file");
        }
        match fs::read(path) {
            Ok(input) => {
                let mut input_copy = input.to_vec();
                input_copy.drain(0..44);
                let mut extractor = FeatureExtractor::new(
                    self.sample_rate,
                    self.samples_per_frame,
                    self.samples_per_shift,
                    self.num_coefficients,
                    self.pre_emphasis_coefficient,
                );
                Ok(self.normalize_features(extractor.process_buffer(&input_copy)))
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
                let mut detected: Option<DetectedWakeword> = None;
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
                        break;
                    }
                    let prev_result = detected.as_ref();
                    if prev_result.is_some() && score < prev_result.unwrap().score {
                        break;
                    }
                    detected = Some(DetectedWakeword {
                        wakeword: wakeword_name.clone(),
                        score,
                    });
                }
                detected
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
