use rubato::{FftFixedInOut, Resampler};

use crate::config::{Endianness, SampleFormat, WavFmt};

/**
 * Encode and convert to wav samples in internal rustpotter format
 */
pub(crate) struct WAVEncoder {
    resampler: Option<FftFixedInOut<f32>>,
    resampler_input_buffer: Option<Vec<Vec<f32>>>,
    resampler_out_buffer: Option<Vec<Vec<f32>>>,
    source_sample_format: SampleFormat,
    source_bits_per_sample: u16,
    source_int_bits_per_sample: u16,
    source_channels: u16,
    source_endianness: Endianness,

    target_bits_per_sample: u16,

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
    pub fn encode(&mut self, buffer: Vec<u8>) -> Vec<f32> {
        match self.source_sample_format {
            SampleFormat::Int => {
                let bits_per_sample = self.source_bits_per_sample;
                let endianness = self.source_endianness.clone();
                self.reencode(encode_int_audio_bytes(buffer, bits_per_sample, endianness))
            }
            SampleFormat::Float => {
                let bits_per_sample = self.source_bits_per_sample;
                let endianness = self.source_endianness.clone();
                self.reencode_float(encode_float_audio_bytes(
                    buffer,
                    bits_per_sample,
                    endianness,
                ))
            }
        }
    }
    pub fn reencode(&mut self, buffer: Vec<i32>) -> Vec<f32> {
        let mono_buffer_with_internal_bit_depth = buffer
            .chunks_exact(self.source_channels as usize)
            .map(|chunk| chunk[0])
            .map(|s| {
                if self.source_int_bits_per_sample == self.target_bits_per_sample {
                    s as f32
                } else if self.source_int_bits_per_sample < self.target_bits_per_sample {
                    (s << (self.target_bits_per_sample - self.source_int_bits_per_sample)) as f32
                } else {
                    (s >> (self.source_int_bits_per_sample - self.target_bits_per_sample)) as f32
                }
            })
            .collect::<Vec<f32>>();
        if self.resampler.is_none() {
            mono_buffer_with_internal_bit_depth
        } else {
            let waves_in = self.resampler_input_buffer.as_mut().unwrap();
            waves_in[0] = mono_buffer_with_internal_bit_depth;
            let waves_out = self.resampler_out_buffer.as_mut().unwrap();
            self.resampler
                .as_mut()
                .unwrap()
                .process_into_buffer(&waves_in, waves_out, None)
                .unwrap();
            waves_out.get(0).unwrap().to_vec()
        }
    }
    pub fn reencode_float(&mut self, audio_chunk: Vec<f32>) -> Vec<f32> {
        self.reencode(
            audio_chunk
                .into_iter()
                .map(|sample| {
                    if sample < 0. {
                        (sample * 0x8000 as f32) as i32
                    } else {
                        (sample * 0x7fff as f32) as i32
                    }
                })
                .collect::<Vec<i32>>(),
        )
    }
    pub fn new(
        input_spec: &WavFmt,
        frame_length_ms: usize,
        target_sample_rate: usize,
        target_bits_per_sample: u16,
    ) -> Result<WAVEncoder, &'static str> {
        let mut input_samples_per_frame = input_spec.sample_rate * frame_length_ms / 1000;
        let mut output_samples_per_frame = input_samples_per_frame;
        let allowed_bits_per_sample = vec![8, 16, 24, 32];
        if !allowed_bits_per_sample.contains(&input_spec.bits_per_sample) {
            Err("Unsupported bits per sample")
        } else if input_spec.sample_rate < 8000 {
            Err("Unsupported sample rate")
        } else if input_spec.channels != 1 && input_spec.channels != 2 {
            Err("Unsupported channel number")
        } else if SampleFormat::Float == input_spec.sample_format && input_spec.bits_per_sample != 32 {
            Err("Bit per sample should be 32 when sample format is float")
        } else {
            let resampler = if input_spec.sample_rate != target_sample_rate {
                let resampler = FftFixedInOut::<f32>::new(
                    input_spec.sample_rate,
                    target_sample_rate,
                    input_samples_per_frame,
                    1,
                )
                .unwrap();
                input_samples_per_frame =
                    resampler.input_frames_next() * input_spec.channels as usize;
                output_samples_per_frame = resampler.output_frames_next();
                Some(resampler)
            } else {
                None
            };
            Ok(WAVEncoder {
                input_samples_per_frame,
                output_samples_per_frame,
                target_bits_per_sample,
                resampler_out_buffer: if resampler.is_some() {
                    Some(resampler.as_ref().unwrap().output_buffer_allocate())
                } else {
                    None
                },
                resampler_input_buffer: if resampler.is_some() {
                    Some(resampler.as_ref().unwrap().input_buffer_allocate())
                } else {
                    None
                },
                resampler,
                source_int_bits_per_sample: if input_spec.sample_format == SampleFormat::Int {
                    input_spec.bits_per_sample
                } else {
                    // float is re-encoded to 16bit int internally
                    16
                },
                source_bits_per_sample: input_spec.bits_per_sample,
                source_sample_format: input_spec.sample_format,
                source_channels: input_spec.channels,
                source_endianness: input_spec.endianness.clone(),
            })
        }
    }
}
fn encode_int_audio_bytes(
    audio_buffer: Vec<u8>,
    bits_per_sample: u16,
    endianness: Endianness,
) -> Vec<i32> {
    let buffer_chunks = audio_buffer.chunks_exact((bits_per_sample / 8) as usize);
    match endianness {
        Endianness::Little => buffer_chunks
            .map(|bytes| match bits_per_sample {
                8 => i8::from_le_bytes([bytes[0]]) as i32,
                16 => i16::from_le_bytes([bytes[0], bytes[1]]) as i32,
                24 => i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]),
                32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0,
            })
            .collect::<Vec<i32>>(),
        Endianness::Big => buffer_chunks
            .map(|bytes| match bits_per_sample {
                8 => i8::from_be_bytes([bytes[0]]) as i32,
                16 => i16::from_be_bytes([bytes[0], bytes[1]]) as i32,
                24 => i32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]),
                32 => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0,
            })
            .collect::<Vec<i32>>(),
        Endianness::Native => buffer_chunks
            .map(|bytes| match bits_per_sample {
                8 => i8::from_ne_bytes([bytes[0]]) as i32,
                16 => i16::from_ne_bytes([bytes[0], bytes[1]]) as i32,
                24 => i32::from_ne_bytes([0, bytes[0], bytes[1], bytes[2]]),
                32 => i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0,
            })
            .collect::<Vec<i32>>(),
    }
}

fn encode_float_audio_bytes(
    audio_buffer: Vec<u8>,
    bits_per_sample: u16,
    endianness: Endianness,
) -> Vec<f32> {
    let buffer_chunks = audio_buffer.chunks_exact((bits_per_sample / 8) as usize);
    let audio_chunk: Vec<f32> = match endianness {
        Endianness::Little => buffer_chunks
            .map(|bytes| match bits_per_sample {
                32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0.,
            })
            .collect::<Vec<f32>>(),
        Endianness::Big => buffer_chunks
            .map(|bytes| match bits_per_sample {
                32 => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0.,
            })
            .collect::<Vec<f32>>(),
        Endianness::Native => buffer_chunks
            .map(|bytes| match bits_per_sample {
                32 => f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _default => 0.,
            })
            .collect::<Vec<f32>>(),
    };
    audio_chunk
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
    let encoder = WAVEncoder::new(&wav_spec, 30, 8000, 16).unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_ne!(input_length, output_length, "input and output have same length");
    assert_eq!(input_length, 960, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
}
