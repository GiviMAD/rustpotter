use crate::{
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        FEATURE_EXTRACTOR_FRAME_SHIFT_MS, FEATURE_EXTRACTOR_NUM_COEFFICIENT,
        FEATURE_EXTRACTOR_PRE_EMPHASIS,
    },
    internal::{
        Dtw, FeatureComparator, FeatureExtractor, FeatureNormalizer, GainNormalizerFilter,
        WAVEncoder,
    },
    wakeword_serde::{DeserializableWakeword, SerializableWakeword},
    Endianness, SampleFormat, WavFmt,
};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, fs::File, io::BufReader, path::Path};

/// Wakeword representation.
#[derive(Serialize, Deserialize)]
pub struct Wakeword {
    pub name: String,
    pub avg_features: Option<Vec<Vec<f32>>>,
    pub samples_features: HashMap<String, Vec<Vec<f32>>>,
    pub threshold: Option<f32>,
    pub avg_threshold: Option<f32>,
    pub rms_level: f32,
    pub enabled: bool,
}
impl SerializableWakeword for Wakeword {}
impl DeserializableWakeword for Wakeword {}
impl Wakeword {
    pub fn new(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        avg_features: Option<Vec<Vec<f32>>>,
        rms_level: f32,
        samples_features: HashMap<String, Vec<Vec<f32>>>,
    ) -> Result<Wakeword, String> {
        if samples_features.is_empty() {
            return Err("Can not create an empty wakeword".to_string());
        }
        Ok(Wakeword {
            name,
            threshold,
            avg_threshold,
            avg_features,
            samples_features,
            rms_level,
            enabled: true,
        })
    }
    pub fn new_from_sample_buffers(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        samples: HashMap<String, Vec<u8>>,
    ) -> Result<Wakeword, String> {
        let mut samples_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut rms_level = 0.;
        for (key, buffer) in samples {
            let mut sample_rms_level = 0.;
            samples_features.insert(
                key,
                compute_sample_features(BufReader::new(buffer.as_slice()), &mut sample_rms_level)?,
            );
            if sample_rms_level > rms_level {
                rms_level = sample_rms_level;
            }
        }
        Wakeword::new(
            name,
            threshold,
            avg_threshold,
            compute_avg_samples_features(&samples_features),
            rms_level,
            samples_features,
        )
    }
    pub fn new_from_sample_files(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        samples: Vec<String>,
    ) -> Result<Wakeword, String> {
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
                compute_sample_features(BufReader::new(file), &mut sample_rms_level)?,
            );
            rms_levels.push(sample_rms_level);
        }
        let rms_level = calc_median(rms_levels);
        Wakeword::new(
            name,
            threshold,
            avg_threshold,
            compute_avg_samples_features(&samples_features),
            rms_level,
            samples_features,
        )
    }
}

pub(crate) fn compute_sample_features<R: std::io::Read>(
    buffer_reader: BufReader<R>,
    out_rms_level: &mut f32,
) -> Result<Vec<Vec<f32>>, String> {
    let wav_reader = WavReader::new(buffer_reader).map_err(|err| err.to_string())?;
    let fmt = WavFmt {
        bits_per_sample: wav_reader.spec().bits_per_sample,
        channels: wav_reader.spec().channels,
        sample_format: wav_reader.spec().sample_format,
        sample_rate: wav_reader.spec().sample_rate as usize,
        endianness: Endianness::Little,
    };
    let mut encoder = WAVEncoder::new(
        &fmt,
        FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        DETECTOR_INTERNAL_SAMPLE_RATE,
    )?;
    let samples_per_frame = encoder.get_output_frame_length();
    let samples_per_shift = (samples_per_frame as f32
        / (FEATURE_EXTRACTOR_FRAME_LENGTH_MS as f32 / FEATURE_EXTRACTOR_FRAME_SHIFT_MS as f32))
        as usize;
    let mut feature_extractor = FeatureExtractor::new(
        DETECTOR_INTERNAL_SAMPLE_RATE,
        samples_per_frame,
        samples_per_shift,
        FEATURE_EXTRACTOR_NUM_COEFFICIENT,
        FEATURE_EXTRACTOR_PRE_EMPHASIS,
    );
    let mut rms_levels: Vec<f32> = Vec::new();
    let encoded_samples = if wav_reader.spec().sample_format == SampleFormat::Int {
        let samples = wav_reader
            .into_samples::<i32>()
            .map(|chunk| *chunk.as_ref().unwrap())
            .collect::<Vec<_>>();
        samples
            .chunks_exact(encoder.get_input_frame_length())
            .map(|buffer| encoder.reencode_int(buffer))
            .map(|encoded_buffer| {
                rms_levels.push(GainNormalizerFilter::get_rms_level(&encoded_buffer));
                encoded_buffer
            })
            .fold(Vec::new(), |mut acc, mut i| {
                acc.append(&mut i);
                acc
            })
    } else {
        let samples = wav_reader
            .into_samples::<f32>()
            .map(|chunk| *chunk.as_ref().unwrap())
            .collect::<Vec<_>>();
        samples
            .chunks_exact(encoder.get_input_frame_length())
            .map(|buffer| encoder.reencode_float(buffer))
            .map(|encoded_buffer| {
                rms_levels.push(GainNormalizerFilter::get_rms_level(&encoded_buffer));
                encoded_buffer
            })
            .fold(Vec::new(), |mut acc, mut i| {
                acc.append(&mut i);
                acc
            })
    };
    if !rms_levels.is_empty() {
        *out_rms_level = calc_median(rms_levels);
    }
    let sample_features = encoded_samples
        .as_slice()
        .chunks_exact(encoder.get_output_frame_length())
        .map(|samples_chunk| feature_extractor.compute_features(samples_chunk))
        .fold(Vec::new() as Vec<Vec<f32>>, |mut acc, feature_matrix| {
            for features in feature_matrix {
                acc.push(features);
            }
            acc
        });
    Ok(FeatureNormalizer::normalize(sample_features))
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
    let mut template_vec = template_values
        .iter()
        .map(|(_, sample)| sample.to_vec())
        .collect::<Vec<Vec<Vec<f32>>>>();
    let mut origin = template_vec.drain(0..1).next().unwrap();
    for frames in template_vec.iter() {
        let mut dtw = Dtw::new(FeatureComparator::calculate_distance);
        dtw.compute_optimal_path(
            &origin.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
        );
        let mut avgs = origin
            .iter()
            .map(|x| x.iter().map(|&y| vec![y]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        dtw.retrieve_optimal_path()
            .unwrap()
            .into_iter()
            .for_each(|[x, y]| {
                frames[y].iter().enumerate().for_each(|(index, feature)| {
                    avgs[x][index].push(*feature);
                })
            });
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

fn calc_median(mut values: Vec<f32>) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    let truncated_mid = values.len() / 2;
    if values.is_empty() {
        0.
    } else if truncated_mid != 0 && values.len() % 2 == 0 {
        values[truncated_mid] + values[truncated_mid - 1]
    } else {
        values[truncated_mid]
    }
}
