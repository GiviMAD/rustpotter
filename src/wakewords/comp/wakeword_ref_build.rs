use std::{cmp::Ordering, collections::HashMap, fs::File, io::BufReader, path::Path};

use crate::{
    mfcc::{MfccAverager, MfccWavFileExtractor},
    WakewordRef,
};

pub trait WakewordRefBuildFromBuffers {
    fn new_from_sample_buffers(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        samples: HashMap<String, Vec<u8>>,
    ) -> Result<WakewordRef, String> {
        let mut samples_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut rms_level = 0.;
        for (key, buffer) in samples {
            let mut sample_rms_level = 0.;
            samples_features.insert(
                key,
                MfccWavFileExtractor::compute_features(
                    BufReader::new(buffer.as_slice()),
                    &mut sample_rms_level,
                )?,
            );
            if sample_rms_level > rms_level {
                rms_level = sample_rms_level;
            }
        }
        WakewordRef::new(
            name,
            threshold,
            avg_threshold,
            compute_avg_samples_features(&samples_features),
            rms_level,
            samples_features,
        )
    }
}
pub trait WakewordRefBuildFromFiles {
    fn new_from_sample_files(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        samples: Vec<String>,
    ) -> Result<WakewordRef, String> {
        let mut samples_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut rms_levels: Vec<f32> = Vec::new();
        for sample_path in samples {
            let path = Path::new(&sample_path);
            if !path.exists() || !path.is_file() {
                return Err("File not found: ".to_owned() + &sample_path);
            }
            let file = match File::open(&sample_path) {
                Ok(it) => it,
                Err(err) => {
                    return Err("Unable to open file ".to_owned()
                        + &sample_path
                        + ": "
                        + &err.to_string())
                }
            };
            let mut sample_rms_level = 0.;
            samples_features.insert(
                String::from(path.file_name().unwrap().to_str().unwrap()),
                MfccWavFileExtractor::compute_features(
                    BufReader::new(file),
                    &mut sample_rms_level,
                )?,
            );
            rms_levels.push(sample_rms_level);
        }
        rms_levels.sort_by(|a, b| a.total_cmp(b));
        let rms_level = rms_levels[rms_levels.len() / 2];
        WakewordRef::new(
            name,
            threshold,
            avg_threshold,
            compute_avg_samples_features(&samples_features),
            rms_level,
            samples_features,
        )
    }
}

fn compute_avg_samples_features(
    templates: &HashMap<String, Vec<Vec<f32>>>,
) -> Option<Vec<Vec<f32>>> {
    if templates.len() <= 1 {
        return None;
    }
    let mut template_values: Vec<_> = templates.iter().collect();
    template_values.sort_by(|a, b| {
        let equality = b.1.len().cmp(&a.1.len());
        if equality == Ordering::Equal {
            a.0.cmp(b.0)
        } else {
            equality
        }
    });
    let template_vec = template_values
        .iter()
        .map(|(_, sample)| sample.to_vec())
        .collect::<Vec<Vec<Vec<f32>>>>();
    MfccAverager::average(template_vec)
}