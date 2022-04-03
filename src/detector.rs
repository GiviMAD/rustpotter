use crate::comparator::FeatureComparator;
use crate::extractor::{FeatureExtractor, FeatureExtractorListener};
use crate::wakeword::{Wakeword, WakewordModel};
use log::{debug, info, warn};
use savefile::{load_file, save_file, save_to_mem};
use std::path::Path;
use std::{collections::HashMap, fs};
pub struct FeatureDetectorBuilder {
    threshold: Option<f32>,
    sample_rate: Option<usize>,
    bit_length: Option<usize>,
    frame_length_ms: Option<usize>,
    frame_shift_ms: Option<usize>,
    num_coefficients: Option<usize>,
    pre_emphasis_coefficient: Option<f32>,
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
        }
    }
    pub fn build<T: FnMut(DetectedWakeword)>(&self, on_event: T) -> FeatureDetector<T> {
        FeatureDetector::new(
            on_event,
            self.get_threshold(),
            self.get_sample_rate(),
            self.get_bit_length(),
            self.get_samples_per_frame(),
            self.get_samples_per_shift(),
            self.get_num_coefficients(),
            self.get_pre_emphasis_coefficient(),
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
}
pub struct FeatureDetector<T> where T: FnMut(DetectedWakeword) {
    on_event: T,
    // options
    threshold: f32,
    sample_rate: usize,
    bit_length: usize,
    samples_per_frame: usize,
    samples_per_shift: usize,
    num_coefficients: usize,
    pre_emphasis_coefficient: f32,
    // state
    buffering: bool,
    min_frames: usize,
    max_frames: usize,
    frames: Vec<Vec<f32>>,
    keywords: HashMap<String, Wakeword>,
    comparator: FeatureComparator,
    result_state: Option<DetectedWakeword>,
    extractor: FeatureExtractor,
    extractor_listener: FeatureExtractorAggregator,
}
impl<'a, T: FnMut(DetectedWakeword)> FeatureExtractorListener for FeatureDetector<T> {
    fn on_features_segment(&mut self, features: Vec<f32>) {
        self.process_features(features);
    }
}
impl<T: FnMut(DetectedWakeword)> FeatureDetector<T> {
    pub fn new(
        on_event: T,
        threshold: f32,
        sample_rate: usize,
        bit_length: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: usize,
        pre_emphasis_coefficient: f32,
    ) -> FeatureDetector<T> {
        let detector = FeatureDetector {
            on_event,
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
            comparator: FeatureComparator::new(None, None),
            extractor: FeatureExtractor::new(
                sample_rate,
                samples_per_frame,
                samples_per_shift,
                num_coefficients,
                pre_emphasis_coefficient,
            ),
            extractor_listener: FeatureExtractorAggregator::new(),
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
                Wakeword::new(
                    enable_average,
                    enabled,
                    threshold,
                ),
            );
        }
        let mut min_frames: usize = 0;
        let mut max_frames: usize = 0;
        for template in templates {
            match self.extract_features_from_file(template) {
                Ok(features) => {
                    let word = self.keywords.get_mut(&keyword).unwrap();
                    word.add_features(features.to_vec());
                    min_frames = if min_frames == 0 { word.get_min_frames() } else { std::cmp::min(min_frames, features.len()) };
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
    pub fn process_bytes(&mut self, buffer: Vec<u8>) {
        self.extractor
            .process_buffer(&buffer, &mut self.extractor_listener);
        let features = self.extractor_listener.get_features();
        if features.len() > 0 {
            for feature in features {
                self.process_features(feature.to_vec());
            }
        }
    }
    pub fn process_pcm_signed(&mut self, buffer: Vec<i16>) {
        self.extractor
            .process_audio(&buffer, &mut self.extractor_listener);
        let features = self.extractor_listener.get_features();
        if features.len() > 0 {
            for feature in features {
                self.process_features(feature.to_vec());
            }
        }
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
                Ok(self.extract_features_from_buffer(input_copy))
            }
            Err(..) => {
                warn!("Can not read file file \"{}\"", file_path);
                Err("Can not read file file")
            }
        }
    }
    fn extract_features_from_buffer(&self, buffer: Vec<u8>) -> Vec<Vec<f32>> {
        let mut aggregator = FeatureExtractorAggregator::new();
        let mut extractor = FeatureExtractor::new(
            self.sample_rate,
            self.samples_per_frame,
            self.samples_per_shift,
            self.num_coefficients,
            self.pre_emphasis_coefficient,
        );
        extractor.process_buffer(&buffer, &mut aggregator);
        let normalized = self.normalize_features(aggregator.get_features());
        normalized
    }
    fn process_features(&mut self, features: Vec<f32>) {
        self.frames.push(features);
        if self.frames.len() >= self.min_frames {
            if self.buffering {
                self.buffering = false;
                println!("Ready");
            }
            self.run_detection();
        }
        if self.frames.len() >= self.max_frames {
            self.frames.drain(0..1);
        }
    }
    fn run_detection(&mut self) {
        let features = self.normalize_features(self.frames.to_vec());
        let result_option = self.get_best_keyword(features);
        match result_option {
            Some(result) => {
                let previous_result = self.result_state.as_ref();
                if self.result_state.is_some() {
                    let detected_wakeword = previous_result.unwrap();
                    if result.wakeword == detected_wakeword.wakeword
                        && result.score < detected_wakeword.score
                    {
                        debug!(
                            "keyword '{}' detected, score {}",
                            result.wakeword,
                            previous_result.unwrap().score
                        );
                        (self.on_event)(detected_wakeword.clone());
                        self.reset();
                        return;
                    }
                }
                self.result_state = Some(result);
            }
            None => {}
        };
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

    fn get_best_keyword(&mut self, features: Vec<Vec<f32>>) -> Option<DetectedWakeword> {
        let mut result: Option<DetectedWakeword> = None;
        for (name, keyword) in &self.keywords {
            if !keyword.is_enabled() {
                continue;
            }
            let threshold = keyword.get_threshold().unwrap_or(self.threshold);
            for template in keyword.get_templates() {
                let mut frames = features.to_vec();
                if frames.len() > template.len() {
                    frames.drain(template.len()..frames.len());
                }
                let score = self.comparator.compare(
                    &template.iter().map(|item| &item[..]).collect::<Vec<_>>(),
                    &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
                );
                if score < threshold {
                    break;
                }
                let prev_result = result.as_ref();
                if prev_result.is_some() && score < prev_result.unwrap().score {
                    break;
                }
                result = Some(DetectedWakeword {
                    wakeword: name.clone(),
                    score,
                });
            }
        }
        return result;
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
        Self { wakeword: self.wakeword.clone(), score: self.score.clone() }
    }
}
struct FeatureExtractorAggregator {
    features: Vec<Vec<f32>>,
}
impl FeatureExtractorAggregator {
    fn new() -> Self {
        FeatureExtractorAggregator {
            features: Vec::new(),
        }
    }
    fn get_features(&mut self) -> Vec<Vec<f32>> {
        self.features.drain(0..self.features.len()).collect()
    }
}
impl FeatureExtractorListener for FeatureExtractorAggregator {
    fn on_features_segment(&mut self, features: Vec<f32>) {
        self.features.push(features);
    }
}
