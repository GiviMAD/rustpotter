use crate::comparator;
use crate::dtw;
pub struct Wakeword {
    averaged_template: Option<Vec<Vec<f32>>>,
    templates: Vec<Vec<Vec<f32>>>,
    enable_average: bool,
    enabled: bool,
    threshold: Option<f32>,
}

#[derive(Savefile)]
pub struct WakewordModel {
    pub keyword: String,
    pub sample_rate: usize,
    pub bit_length: usize,
    pub channels: usize,
    pub samples_per_frame: usize,
    pub samples_per_shift: usize,
    pub num_coefficients: usize,
    pub pre_emphasis_coefficient: f32,
    threshold: Option<f32>,
    templates: Vec<Vec<Vec<f32>>>,
}
impl WakewordModel {
    pub fn new(
        keyword: String,
        templates: Vec<Vec<Vec<f32>>>,
        sample_rate: usize,
        bit_length: usize,
        channels: usize,
        samples_per_frame: usize,
        samples_per_shift: usize,
        num_coefficients: usize,
        pre_emphasis_coefficient: f32,
        threshold: Option<f32>,
    ) -> Self {
        WakewordModel {
            keyword,
            templates,
            sample_rate,
            bit_length,
            channels,
            samples_per_frame,
            samples_per_shift,
            num_coefficients,
            pre_emphasis_coefficient,
            threshold,
        }
    }
}
impl Wakeword {
    pub fn from_model(
        model: WakewordModel,
        enable_average: bool,
        enabled: bool,
    ) -> Wakeword {
        let mut wakeword = Wakeword {
            averaged_template: None,
            templates: model.templates.to_vec(),
            enable_average: enable_average,
            enabled: enabled,
            threshold: model.threshold,
        };
        if enable_average {
            wakeword.average_templates();
        }
        wakeword
    }
    pub fn new(
        enable_average: bool,
        enabled: bool,
        threshold: Option<f32>,
    ) -> Wakeword {
        Wakeword {
            enable_average,
            enabled,
            threshold,
            templates: vec![],
            averaged_template: None,
        }
    }
    pub fn get_min_frames(&self) -> usize {
        self.get_templates()
            .iter()
            .map(|item| item.len())
            .min()
            .expect("Unable to get min frames for wakeword")
    }
    pub fn get_max_frames(&self) -> usize {
        self.get_templates()
            .iter()
            .map(|item| item.len())
            .max()
            .expect("Unable to get min frames for wakeword")
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn get_threshold(&self) -> Option<f32> {
        self.threshold
    }
    pub fn get_templates(&self) -> Vec<Vec<Vec<f32>>> {
        if !self.enable_average {
            self.templates.to_vec()
        } else {
            vec![self.averaged_template.as_ref().unwrap().to_vec()]
        }
    }
    pub fn add_features(&mut self, features: Vec<Vec<f32>>) {
        self.templates.push(features);
        if self.enable_average {
            self.average_templates()
        }
    }
    fn average_templates(&mut self) {
        if self.templates.len() <= 1  {
            return;
        }
        self.templates
            .sort_by(|a, b| a.len().partial_cmp(&b.len()).unwrap());
        let mut origin = self.templates[0].to_vec();
        for i in 1..self.templates.len() {
            let frames = self.templates[i].to_vec();

            let mut dtw = dtw::new(comparator::calculate_distance);

            let _ = dtw.compute_optimal_path(
                &origin.iter().map(|item| &item[..]).collect::<Vec<_>>(),
                &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            );
            let mut avgs = origin
                .iter()
                .map(|x| x.iter().map(|&y| vec![y]).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for tuple in dtw.retrieve_optimal_path().unwrap() {
                for index in 0..frames[tuple[1]].len() {
                    let feature = frames[tuple[1]][i];
                    avgs[tuple[0]][index].push(feature);
                }
            }
            origin = avgs
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|feature_group| {
                            feature_group.to_vec().iter().sum::<f32>() / feature_group.len() as f32
                        })
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<_>>();
        }
        self.averaged_template = Some(origin);
    }
}
