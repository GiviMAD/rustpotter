use crate::comparator;
use crate::dtw;
pub struct Wakeword {
    averaged_template: Option<Vec<Vec<f32>>>,
    templates: Vec<Vec<Vec<f32>>>,
    enabled: bool,
    threshold: Option<f32>,
    averaged_threshold: Option<f32>,
}

#[derive(Savefile)]
pub struct WakewordModel {
    pub name: String,
    threshold: Option<f32>,
    averaged_threshold: Option<f32>,
    templates: Vec<Vec<Vec<f32>>>,
}
impl WakewordModel {
    pub fn new(
        name: String,
        templates: Vec<Vec<Vec<f32>>>,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
    ) -> Self {
        WakewordModel {
            name,
            templates,
            averaged_threshold,
            threshold,
        }
    }
}
impl Wakeword {
    pub fn from_model(model: WakewordModel, enabled: bool) -> Wakeword {
        Wakeword {
            averaged_template: average_templates(&mut model.templates.to_vec()),
            templates: model.templates.to_vec(),
            enabled: enabled,
            threshold: model.threshold,
            averaged_threshold: model.averaged_threshold,
        }
    }
    pub fn new(enabled: bool, averaged_threshold: Option<f32>, threshold: Option<f32>) -> Wakeword {
        Wakeword {
            enabled,
            threshold,
            averaged_threshold: if averaged_threshold.is_some() {
                averaged_threshold
            } else if threshold.is_some() {
                Some(threshold.unwrap() / 2.)
            } else {
                None
            },
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
    pub fn get_averaged_threshold(&self) -> Option<f32> {
        self.averaged_threshold
    }
    pub fn get_averaged_template(&self) -> Option<Vec<Vec<f32>>> {
        if self.averaged_template.is_some() {
            Some(self.averaged_template.as_ref().unwrap().to_vec())
        } else {
            None
        }
    }
    pub fn get_templates(&self) -> Vec<Vec<Vec<f32>>> {
        self.templates.to_vec()
    }
    pub fn add_features(&mut self, features: Vec<Vec<f32>>) {
        self.templates.push(features);
        self.averaged_template = average_templates(&mut self.templates);
    }
    pub fn prioritize_template(&mut self, index: usize) {
        self.templates.rotate_right(1);
        self.templates.swap(index, 0);
    }
}
fn average_templates(templates: &mut [Vec<Vec<f32>>]) -> Option<Vec<Vec<f32>>> {
    if templates.len() <= 1 {
        return None;
    }
    templates.sort_by(|a, b| a.len().partial_cmp(&b.len()).unwrap());
    let mut origin = templates[0].to_vec();
    for i in 1..templates.len() {
        let frames = templates[i].to_vec();

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
    Some(origin)
}
