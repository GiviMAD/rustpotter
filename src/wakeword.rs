use std::{cmp::Ordering, collections::HashMap, fs::File, io::BufReader, path::Path};

use ciborium::{de, ser};
use hound::WavReader;

use crate::{
    internal::{
        Dtw, FeatureComparator, FeatureExtractor, FeatureNormalizer, GainNormalizerFilter,
        WAVEncoder,
    },
    Endianness, SampleFormat, WavFmt, DETECTOR_INTERNAL_BIT_DEPTH, DETECTOR_INTERNAL_SAMPLE_RATE,
    FEATURE_EXTRACTOR_FRAME_LENGTH_MS, FEATURE_EXTRACTOR_FRAME_SHIFT_MS,
    FEATURE_EXTRACTOR_NUM_COEFFICIENT, FEATURE_EXTRACTOR_PRE_EMPHASIS,
};
use serde::{Deserialize, Serialize};

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
impl Wakeword {
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(it) => it,
            Err(err) => {
                return Err(String::from(
                    "Unable to open file ".to_owned() + path + ": " + &err.to_string(),
                ))
            }
        };
        ser::into_writer(self, &mut file).map_err(|err| err.to_string())?;
        Ok(())
    }
    pub fn save_to_buffer(&self) -> Result<Vec<u8>, String> {
        let mut bytes: Vec<u8> = Vec::new();
        ser::into_writer(self, &mut bytes).map_err(|err| err.to_string())?;
        Ok(bytes)
    }
    pub fn load_from_file(path: &str) -> Result<Wakeword, String> {
        let file = match File::open(path) {
            Ok(it) => it,
            Err(err) => {
                return Err(String::from(
                    "Unable to open file ".to_owned() + path + ": " + &err.to_string(),
                ))
            }
        };
        let reader = BufReader::new(file);
        let wakeword: Wakeword = de::from_reader(reader).map_err(|err| err.to_string())?;
        Ok(wakeword)
    }
    pub fn load_from_buffer(buffer: &[u8]) -> Result<Wakeword, String> {
        let reader = BufReader::new(buffer);
        let wakeword: Wakeword = de::from_reader(reader).map_err(|err| err.to_string())?;
        Ok(wakeword)
    }
    pub fn new(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        avg_features: Option<Vec<Vec<f32>>>,
        rms_level: f32,
        samples_features: HashMap<String, Vec<Vec<f32>>>,
    ) -> Result<Wakeword, String> {
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
        let mut rms_level = 0.;
        for sample_path in samples {
            let path = Path::new(&sample_path);
            if !path.exists() || !path.is_file() {
                return Err(String::from("File not found: ".to_owned() + &sample_path));
            }
            let file = match File::open(&sample_path) {
                Ok(it) => it,
                Err(err) => {
                    return Err(String::from(
                        "Unable to open file ".to_owned() + &sample_path + ": " + &err.to_string(),
                    ))
                }
            };
            let mut sample_rms_level = 0.;
            samples_features.insert(
                String::from(path.file_name().unwrap().to_str().unwrap()),
                compute_sample_features(BufReader::new(file), &mut sample_rms_level)?,
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
}
fn compute_sample_features<R: std::io::Read>(
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
        DETECTOR_INTERNAL_BIT_DEPTH,
    )?;
    let samples_per_frame = encoder.get_output_frame_length();
    let samples_per_shift = (samples_per_frame as f32
        / (FEATURE_EXTRACTOR_FRAME_LENGTH_MS as f32 / FEATURE_EXTRACTOR_FRAME_SHIFT_MS as f32)
            as f32) as usize;
    let mut feature_extractor = FeatureExtractor::new(
        DETECTOR_INTERNAL_SAMPLE_RATE,
        samples_per_frame,
        samples_per_shift,
        FEATURE_EXTRACTOR_NUM_COEFFICIENT,
        FEATURE_EXTRACTOR_PRE_EMPHASIS,
    );
    // used to calculate measure wakeword samples loudness
    let gain_normalizer_filter = GainNormalizerFilter::new();
    let encoded_samples = if wav_reader.spec().sample_format == SampleFormat::Int {
        let samples = wav_reader
            .into_samples::<i32>()
            .map(|chunk| *chunk.as_ref().unwrap())
            .collect::<Vec<_>>();
        samples
            .chunks_exact(encoder.get_input_frame_length())
            .map(|buffer| encoder.reencode_int(buffer))
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
            .fold(Vec::new(), |mut acc, mut i| {
                acc.append(&mut i);
                acc
            })
    };
    *out_rms_level = gain_normalizer_filter.get_rms_level(&encoded_samples);
    let sample_features = encoded_samples
        .as_slice()
        .chunks_exact(encoder.get_output_frame_length())
        .map(|samples_chunk| feature_extractor.compute_features(&samples_chunk))
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
    let mut template_values: Vec<_> = templates.into_iter().collect();
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
    for (i, frames) in template_vec.iter().enumerate() {
        let mut dtw = Dtw::new(FeatureComparator::calculate_distance);

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
