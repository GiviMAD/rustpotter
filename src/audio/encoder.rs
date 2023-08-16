use rubato::{FftFixedInOut, Resampler};

use crate::config::{Endianness, SampleFormat, WavFmt};

/**
 * Encode and convert to wav samples in internal rustpotter format
 */
pub struct WAVEncoder {
    resampler: Option<FftFixedInOut<f32>>,
    resampler_input_buffer: Option<Vec<Vec<f32>>>,
    resampler_out_buffer: Option<Vec<Vec<f32>>>,
    source_sample_format: SampleFormat,
    source_bits_per_sample: u16,
    source_channels: u16,
    source_endianness: Endianness,
    input_samples_per_frame: usize,
    output_samples_per_frame: usize,
}
impl WAVEncoder {
    pub fn get_input_frame_length(&self) -> usize {
        self.input_samples_per_frame
    }
    pub fn get_output_frame_length(&self) -> usize {
        self.output_samples_per_frame
    }
    pub fn get_input_byte_length(&self) -> usize {
        self.get_input_frame_length() * (self.source_bits_per_sample as usize / 8)
    }
    pub fn encode_and_resample(&mut self, buffer: &[u8]) -> Vec<f32> {
        let float_samples = if self.source_sample_format == SampleFormat::Int {
            match self.source_bits_per_sample {
                8 => encode_audio_bytes::<i8>(buffer, self.source_endianness),
                16 => encode_audio_bytes::<i16>(buffer, self.source_endianness),
                32 => encode_audio_bytes::<i32>(buffer, self.source_endianness),
                _ => Vec::new(),
            }
        } else {
            encode_audio_bytes::<f32>(buffer, self.source_endianness)
        };
        self.reencode_to_mono_with_sample_rate(float_samples)
    }
    pub fn rencode_and_resample<T: SampleType>(&mut self, buffer: Vec<T>) -> Vec<f32> {
        self.reencode_to_mono_with_sample_rate(
            buffer.into_iter().map(T::into_f32).collect::<Vec<f32>>(),
        )
    }
    fn reencode_to_mono_with_sample_rate(&mut self, buffer: Vec<f32>) -> Vec<f32> {
        let mono_buffer = if self.source_channels != 1 {
            buffer
                .chunks_exact(self.source_channels as usize)
                .map(|chunk| chunk[0])
                .collect::<Vec<f32>>()
        } else {
            buffer
        };
        if self.resampler.is_none() {
            mono_buffer
        } else {
            let waves_in = self.resampler_input_buffer.as_mut().unwrap();
            waves_in[0] = mono_buffer;
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            self.resampler
                .as_mut()
                .unwrap()
                .process_into_buffer(waves_in, waves_out, None)
                .unwrap();
            waves_out.get(0).unwrap().to_vec()
        }
    }
    pub fn new(
        input_spec: &WavFmt,
        frame_length_ms: usize,
        target_sample_rate: usize,
    ) -> Result<WAVEncoder, &'static str> {
        let mut input_samples_per_frame =
            (input_spec.sample_rate * frame_length_ms / 1000) * input_spec.channels as usize;
        let output_samples_per_frame = target_sample_rate * frame_length_ms / 1000;
        let allowed_bits_per_sample = vec![8, 16, 24, 32];
        if !allowed_bits_per_sample.contains(&input_spec.bits_per_sample) {
            Err("Unsupported bits per sample")
        } else if input_spec.sample_rate < 8000 {
            Err("Unsupported sample rate")
        } else if input_spec.channels != 1 && input_spec.channels != 2 {
            Err("Unsupported channel number")
        } else if SampleFormat::Float == input_spec.sample_format
            && input_spec.bits_per_sample != 32
        {
            Err("Bit per sample should be 32 when sample format is float")
        } else {
            let resampler = if input_spec.sample_rate != target_sample_rate {
                let resampler = FftFixedInOut::<f32>::new(
                    input_spec.sample_rate,
                    target_sample_rate,
                    output_samples_per_frame,
                    1,
                )
                .map_err(|_| "Unsupported sample rate, unable to initialize the resampler")?;
                input_samples_per_frame =
                    resampler.input_frames_next() * input_spec.channels as usize;
                Some(resampler)
            } else {
                None
            };
            Ok(WAVEncoder {
                input_samples_per_frame,
                output_samples_per_frame,
                resampler_out_buffer: if resampler.is_some() {
                    Some(resampler.as_ref().unwrap().output_buffer_allocate(true))
                } else {
                    None
                },
                resampler_input_buffer: if resampler.is_some() {
                    Some(resampler.as_ref().unwrap().input_buffer_allocate(false))
                } else {
                    None
                },
                resampler,
                source_bits_per_sample: input_spec.bits_per_sample,
                source_sample_format: input_spec.sample_format,
                source_channels: input_spec.channels,
                source_endianness: input_spec.endianness,
            })
        }
    }
}

pub trait SampleType:
    Sized + Copy + std::cmp::PartialOrd + std::fmt::Display + 'static + Send
{
    fn get_byte_size() -> usize;
    fn get_zero() -> Self;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn from_be_bytes(bytes: &[u8]) -> Self;
    fn from_ne_bytes(bytes: &[u8]) -> Self;
    fn into_f32(self) -> f32;
}

macro_rules! with_sample_type {
    ($ty:ty, $byte_size:literal, $to_f32:expr, $from_le_bytes:expr, $from_be_bytes:expr, $from_ne_bytes:expr, $zero:literal) => {
        impl SampleType for $ty {
            fn get_byte_size() -> usize {
                $byte_size
            }
            fn get_zero() -> Self {
                $zero
            }
            fn into_f32(self) -> f32 {
                $to_f32(self)
            }
            fn from_le_bytes(bytes: &[u8]) -> Self {
                $from_le_bytes(bytes)
            }
            fn from_be_bytes(bytes: &[u8]) -> Self {
                $from_be_bytes(bytes)
            }
            fn from_ne_bytes(bytes: &[u8]) -> Self {
                $from_ne_bytes(bytes)
            }
        }
    };
}
with_sample_type!(
    i8,
    1,
    |v: i8| v as f32 / (i8::MAX as f32),
    |bytes: &[u8]| i8::from_le_bytes([bytes[0]]),
    |bytes: &[u8]| i8::from_be_bytes([bytes[0]]),
    |bytes: &[u8]| i8::from_ne_bytes([bytes[0]]),
    0
);
with_sample_type!(
    i16,
    2,
    |v: i16| v as f32 / (i16::MAX as f32),
    |bytes: &[u8]| i16::from_le_bytes([bytes[0], bytes[1]]),
    |bytes: &[u8]| i16::from_be_bytes([bytes[0], bytes[1]]),
    |bytes: &[u8]| i16::from_ne_bytes([bytes[0], bytes[1]]),
    0
);
with_sample_type!(
    i32,
    4,
    |v: i32| v as f32 / (i32::MAX as f32),
    |bytes: &[u8]| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    0
);
with_sample_type!(
    f32,
    4,
    |v: f32| v,
    |bytes: &[u8]| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    0.
);

fn encode_audio_bytes<T: SampleType>(audio_buffer: &[u8], endianness: Endianness) -> Vec<f32> {
    let buffer_chunks = audio_buffer.chunks_exact(T::get_byte_size());
    match endianness {
        Endianness::Little => buffer_chunks
            .map(T::from_le_bytes)
            .collect::<Vec<T>>(),
        Endianness::Big => buffer_chunks
            .map(T::from_be_bytes)
            .collect::<Vec<T>>(),
        Endianness::Native => buffer_chunks
            .map(T::from_ne_bytes)
            .collect::<Vec<T>>(),
    }
    .into_iter()
    .map(T::into_f32)
    .collect()
}

#[test]
fn it_returns_correct_samples_per_frame() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let file = std::fs::File::open(dir.to_owned() + "/tests/resources/oye_casa_g_1.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(file)).unwrap();
    let sample_rate = wav_reader.spec().sample_rate as usize;
    let sample_format = wav_reader.spec().sample_format;
    let bits_per_sample = wav_reader.spec().bits_per_sample;
    let channels = wav_reader.spec().channels;
    let wav_spec = WavFmt {
        sample_rate,
        sample_format,
        bits_per_sample,
        channels,
        endianness: Endianness::Little,
    };
    let encoder = WAVEncoder::new(
        &wav_spec,
        crate::constants::FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_eq!(
        input_length, output_length,
        "input and output have same length"
    );
    assert_eq!(input_length, 480, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
}

#[test]
fn reencode_wav_with_different_format() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let i16_samples_file =
        std::fs::File::open(dir.to_owned() + "/tests/resources/oye_casa_g_1.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(i16_samples_file)).unwrap();
    let wav_spec = WavFmt {
        sample_rate: wav_reader.spec().sample_rate as usize,
        sample_format: wav_reader.spec().sample_format,
        bits_per_sample: wav_reader.spec().bits_per_sample,
        channels: wav_reader.spec().channels,
        endianness: Endianness::Little,
    };
    println!("{:?}", wav_spec);
    let mut encoder = WAVEncoder::new(
        &wav_spec,
        crate::constants::FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_eq!(
        input_length, output_length,
        "input and output have same length"
    );
    assert_eq!(input_length, 480, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
    let samples = wav_reader
        .into_samples::<i16>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let internal_spec = hound::WavSpec {
        sample_rate: crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        channels: 1,
    };
    let mut writer = hound::WavWriter::create(
        dir.to_owned() + "/tests/resources/oye_casa_g_1_f32.wav",
        internal_spec,
    )
    .unwrap();
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.rencode_and_resample::<i16>(chuck.to_vec()))
        .for_each(|reencoded_chunk| {
            for sample in reencoded_chunk {
                writer.write_sample(sample).ok();
            }
        });
    writer.finalize().expect("Unable to save file");
}
