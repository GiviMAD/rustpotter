#[cfg(feature = "build")]
use crate::comparator;
#[cfg(feature = "build")]
use crate::dtw;

pub const WAKEWORD_MODEL_VERSION: u32 = 0;
#[derive(Savefile)]
pub struct WakewordModel {
    name: String,
    threshold: Option<f32>,
    averaged_threshold: Option<f32>,
    averaged_template: Option<Vec<Vec<f32>>>,
    templates: Vec<(String, Vec<Vec<f32>>)>,
}
impl WakewordModel {
    #[cfg(feature = "build")]
    pub fn new(
        name: &str,
        averaged_template: Option<Vec<Vec<f32>>>,
        templates: Vec<WakewordTemplate>,
        averaged_threshold: Option<f32>,
        threshold: Option<f32>,
    ) -> Self {
        WakewordModel {
            name: String::from(name),
            templates: templates
                .into_iter()
                .map(|item| (item._name, item.template))
                .collect(),
            averaged_template,
            averaged_threshold,
            threshold,
        }
    }
    #[cfg(feature = "build")]
    pub fn from_wakeword(name: &str, wakeword: &Wakeword) -> Self {
        WakewordModel::new(
            name,
            wakeword.get_averaged_template(),
            wakeword.get_templates(),
            wakeword.get_averaged_threshold(),
            wakeword.get_threshold(),
        )
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}
#[derive(Clone)]
pub struct WakewordTemplate {
    _name: String,
    template: Vec<Vec<f32>>,
}

impl WakewordTemplate {
    pub fn _get_name(&self) -> &str {
        &self._name
    }
    pub fn get_template(&self) -> &[Vec<f32>] {
        &self.template
    }
}
pub struct Wakeword {
    averaged_template: Option<Vec<Vec<f32>>>,
    templates: Vec<WakewordTemplate>,
    enabled: bool,
    threshold: Option<f32>,
    averaged_threshold: Option<f32>,
}
impl Wakeword {
    pub fn from_model(model: WakewordModel, enabled: bool) -> Wakeword {
        Wakeword {
            averaged_template: model.averaged_template,
            templates: model
                .templates
                .into_iter()
                .map(|(name, template)| WakewordTemplate { _name: name, template })
                .collect(),
            enabled,
            threshold: model.threshold,
            averaged_threshold: model.averaged_threshold,
        }
    }
    #[cfg(feature = "build")]
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
            templates: Vec::new(),
            averaged_template: None,
        }
    }
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = Some(threshold);
    }
    pub fn set_averaged_threshold(&mut self, averaged_threshold: f32) {
        self.averaged_threshold = Some(averaged_threshold);
    }
    pub fn get_min_frames(&self) -> usize {
        self.get_templates()
            .iter()
            .map(|item| item.template.len())
            .min()
            .unwrap_or(9999)
    }
    pub fn get_max_frames(&self) -> usize {
        self.get_templates()
            .iter()
            .map(|item| item.template.len())
            .max()
            .unwrap_or(0)
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
    pub fn get_templates(&self) -> Vec<WakewordTemplate> {
        self.templates.clone()
    }
    #[cfg(feature = "build")]
    pub fn add_templates_features(&mut self, templates: Vec<(String, Vec<Vec<f32>>)>) {
        templates
            .into_iter()
            .for_each(|(name, template)| self.templates.push(WakewordTemplate { _name: name, template }));
        self.averaged_template = average_templates(&self.templates);
    }
    pub fn prioritize_template(&mut self, index: usize) {
        self.templates.rotate_right(1);
        self.templates.swap(index, 0);
    }
}
#[cfg(feature = "build")]
fn average_templates(templates: &[WakewordTemplate]) -> Option<Vec<Vec<f32>>> {
    if templates.len() <= 1 {
        return None;
    }
    let mut template_vec = templates.to_vec();
    template_vec.sort_by(|a, b| a.template.len().partial_cmp(&b.template.len()).unwrap());
    let mut origin = template_vec[0].template.to_vec();
    for (i, template_vec) in templates.iter().enumerate().skip(1) {
        let frames = template_vec.template.to_vec();

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
