use std::io::BufReader;

use hound::WavReader;

use crate::{
    audio::{GainNormalizerFilter, WAVEncoder},
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        FEATURE_EXTRACTOR_FRAME_SHIFT_MS, FEATURE_EXTRACTOR_NUM_COEFFICIENT,
        FEATURE_EXTRACTOR_PRE_EMPHASIS,
    },
    Endianness, SampleFormat, WavFmt,
};

use super::{MfccExtractor, MfccNormalizer};

pub(crate) struct MfccWavFileExtractor {}
impl MfccWavFileExtractor {
    pub(crate) fn compute_features<R: std::io::Read>(
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
        let mut feature_extractor = MfccExtractor::new(
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
            rms_levels.sort_by(|a, b| a.total_cmp(b));
            let rms_level = rms_levels[rms_levels.len() / 2];
            *out_rms_level = rms_level;
        }
        let sample_mfccs = encoded_samples
            .as_slice()
            .chunks_exact(encoder.get_output_frame_length())
            .map(|samples_chunk| feature_extractor.compute(samples_chunk))
            .fold(Vec::new() as Vec<Vec<f32>>, |mut acc, feature_matrix| {
                for features in feature_matrix {
                    acc.push(features);
                }
                acc
            });
        Ok(MfccNormalizer::normalize(sample_mfccs))
    }
}
