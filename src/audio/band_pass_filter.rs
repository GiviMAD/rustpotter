use std::f32::consts::PI;

use crate::{constants::DETECTOR_INTERNAL_SAMPLE_RATE, BandPassConfig};

pub struct BandPassFilter {
    // options
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,
    // state
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}
impl BandPassFilter {
    pub fn filter(&mut self, signal: &mut [f32]) {
        for sample in signal.iter_mut() {
            let x = *sample;
            *sample = self.a0 * x + self.a1 * self.x1 + self.a2 * self.x2
                - self.b1 * self.y1
                - self.b2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = *sample;
        }
    }
    pub fn new(sample_rate: f32, low_cutoff: f32, high_cutoff: f32) -> Self {
        let omega_low = 2.0 * PI * low_cutoff / sample_rate;
        let omega_high = 2.0 * PI * high_cutoff / sample_rate;
        let cos_omega_low = omega_low.cos();
        let cos_omega_high = omega_high.cos();
        let alpha_low = omega_low.sin() / 2.0;
        let alpha_high = omega_high.sin() / 2.0;
        let a0 = 1.0 / (1.0 + alpha_high - alpha_low);
        let a1 = -2.0 * cos_omega_low * a0;
        let a2 = (1.0 - alpha_high - alpha_low) * a0;
        let b1 = -2.0 * cos_omega_high * a0;
        let b2 = (1.0 - alpha_high + alpha_low) * a0;
        Self {
            a0,
            a1,
            a2,
            b1,
            b2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}
impl From<&BandPassConfig> for Option<BandPassFilter> {
    fn from(config: &BandPassConfig) -> Self {
        if config.enabled {
            Some(BandPassFilter::new(
                DETECTOR_INTERNAL_SAMPLE_RATE as f32,
                config.low_cutoff,
                config.high_cutoff,
            ))
        } else {
            None
        }
    }
}
#[test]
fn filter_audio() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_file =
        std::fs::File::open(dir.to_owned() + "/tests/resources/real_sample.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(sample_file)).unwrap();
    let mut encoder = crate::audio::WAVEncoder::new(
        &wav_reader.spec().try_into().unwrap(),
        crate::constants::MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_ne!(
        input_length, output_length,
        "input and output not have same length"
    );
    assert_eq!(input_length, 1440, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
    let samples = wav_reader
        .into_samples::<f32>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let internal_spec = hound::WavSpec {
        sample_rate: crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        channels: 1,
    };
    let mut writer = hound::WavWriter::create(
        dir.to_owned() + "/tests/resources/band-pass_example.wav",
        internal_spec,
    )
    .unwrap();
    let mut filter = BandPassFilter::new(
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as f32,
        80.,
        400.,
    );
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.rencode_and_resample::<f32>(chuck.into()))
        .map(|mut chunk| {
            filter.filter(&mut chunk);
            chunk
        })
        .for_each(|encoded_chunk| {
            for sample in encoded_chunk {
                writer.write_sample(sample).ok();
            }
        });
    writer.finalize().expect("Unable to save file");
}

#[test]
fn filter_gain_normalized_audio() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_file =
        std::fs::File::open(dir.to_owned() + "/tests/resources/real_sample.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(sample_file)).unwrap();
    let mut encoder = crate::audio::WAVEncoder::new(
        &wav_reader.spec().try_into().unwrap(),
        crate::constants::MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_ne!(
        input_length, output_length,
        "input and output not have same length"
    );
    assert_eq!(input_length, 1440, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
    let samples = wav_reader
        .into_samples::<f32>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let internal_spec = hound::WavSpec {
        sample_rate: crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        channels: 1,
    };
    let mut writer = hound::WavWriter::create(
        dir.to_owned() + "/tests/resources/gain_normalized_band-pass_example.wav",
        internal_spec,
    )
    .unwrap();
    let mut gain_filter = crate::audio::GainNormalizerFilter::new(0.1, 1., Some(0.003));
    let mut filter = BandPassFilter::new(
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as f32,
        80.,
        400.,
    );
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.rencode_and_resample::<f32>(chuck.into()))
        .map(|mut chunk| {
            let rms_level = crate::audio::GainNormalizerFilter::get_rms_level(&chunk);
            gain_filter.filter(&mut chunk, rms_level);
            filter.filter(&mut chunk);
            chunk
        })
        .for_each(|encoded_chunk| {
            for sample in encoded_chunk {
                writer.write_sample(sample).ok();
            }
        });
    writer.finalize().expect("Unable to save file");
}
