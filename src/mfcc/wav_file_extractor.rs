use std::io::BufReader;

use hound::{WavReader, WavSpec};

use crate::{
    audio::{AudioEncoder, GainNormalizerFilter},
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        MFCCS_EXTRACTOR_FRAME_SHIFT_MS, MFCCS_EXTRACTOR_PRE_EMPHASIS,
    },
    AudioFmt, Endianness, Sample, SampleFormat,
};

use super::{MfccExtractor, MfccNormalizer};

pub(crate) struct MfccWavFileExtractor {}
impl MfccWavFileExtractor {
    pub(crate) fn compute_mfccs<R: std::io::Read>(
        buffer_reader: BufReader<R>,
        out_rms_level: &mut f32,
        mfcc_size: u16,
    ) -> Result<Vec<Vec<f32>>, String> {
        let wav_reader = WavReader::new(buffer_reader).map_err(|err| err.to_string())?;
        let fmt = wav_reader.spec().try_into()?;
        let mut encoder = AudioEncoder::new(
            &fmt,
            MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )?;
        let samples_per_frame = encoder.get_output_frame_length();
        let samples_per_shift = (samples_per_frame as f32
            / (MFCCS_EXTRACTOR_FRAME_LENGTH_MS as f32 / MFCCS_EXTRACTOR_FRAME_SHIFT_MS as f32))
            as usize;
        let mut mfcc_extractor = MfccExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
            samples_per_frame,
            samples_per_shift,
            mfcc_size + 1, // first coefficient is dropped
            MFCCS_EXTRACTOR_PRE_EMPHASIS,
        );
        let mut rms_levels: Vec<f32> = Vec::new();
        let encoded_samples = match fmt.sample_format {
            SampleFormat::I8 => encode_samples::<R, i8>(wav_reader, &mut encoder, &mut rms_levels),
            SampleFormat::I16 => {
                encode_samples::<R, i16>(wav_reader, &mut encoder, &mut rms_levels)
            }
            SampleFormat::I32 => {
                encode_samples::<R, i32>(wav_reader, &mut encoder, &mut rms_levels)
            }
            SampleFormat::F32 => {
                encode_samples::<R, f32>(wav_reader, &mut encoder, &mut rms_levels)
            }
        };
        if !rms_levels.is_empty() {
            rms_levels.sort_by(|a, b| a.total_cmp(b));
            let rms_level = rms_levels[rms_levels.len() / 2];
            *out_rms_level = rms_level;
        }
        let sample_mfccs = encoded_samples
            .as_slice()
            .chunks_exact(encoder.get_output_frame_length())
            .map(|samples_chunk| mfcc_extractor.compute(samples_chunk))
            .fold(Vec::new() as Vec<Vec<f32>>, |mut acc, mfcc_matrix| {
                mfcc_matrix.into_iter().for_each(|mfccs| acc.push(mfccs));
                acc
            });
        Ok(MfccNormalizer::normalize(sample_mfccs))
    }
}

fn encode_samples<R: std::io::Read, S: hound::Sample + Sample>(
    wav_reader: WavReader<BufReader<R>>,
    encoder: &mut AudioEncoder,
    rms_levels: &mut Vec<f32>,
) -> Vec<f32> {
    let samples = wav_reader
        .into_samples::<S>()
        .map(|chunk| chunk.unwrap())
        .collect::<Vec<_>>();
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.rencode_and_resample::<S>(chuck.into()))
        .map(|encoded_buffer| {
            rms_levels.push(GainNormalizerFilter::get_rms_level(&encoded_buffer));
            encoded_buffer
        })
        .fold(Vec::new(), |mut acc, mut i| {
            acc.append(&mut i);
            acc
        })
}

impl TryFrom<WavSpec> for AudioFmt {
    type Error = String;

    fn try_from(spec: WavSpec) -> Result<Self, Self::Error> {
        let sample_format = match &spec.sample_format {
            hound::SampleFormat::Int => SampleFormat::int_of_size(spec.bits_per_sample),
            hound::SampleFormat::Float => SampleFormat::float_of_size(spec.bits_per_sample),
        };
        if let Some(sample_format) = sample_format {
            Ok(AudioFmt {
                channels: spec.channels,
                sample_format,
                sample_rate: spec.sample_rate as usize,
                endianness: Endianness::Little,
            })
        } else {
            Err("Unsupported wav format".to_string())
        }
    }
}
