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
    pub fn encode(&mut self, buffer: &[u8]) -> Vec<f32> {
        match self.source_sample_format {
            SampleFormat::Int => {
                let bits_per_sample = self.source_bits_per_sample;
                let endianness = self.source_endianness.clone();
                self.reencode_int(&encode_int_audio_bytes(buffer, bits_per_sample, endianness))
            }
            SampleFormat::Float => {
                let bits_per_sample = self.source_bits_per_sample;
                let endianness = self.source_endianness.clone();
                self.reencode_float(&encode_float_audio_bytes(
                    buffer,
                    bits_per_sample,
                    endianness,
                ))
            }
        }
    }
    pub fn reencode_int(&mut self, buffer: &[i32]) -> Vec<f32> {
        self.reencode_to_mono_with_sample_rate(
            &buffer
                .chunks_exact(self.source_channels as usize)
                .map(|chunk| chunk[0])
                .map(|s| {
                    if self.source_int_bits_per_sample == self.target_bits_per_sample {
                        s as f32
                    } else if self.source_int_bits_per_sample < self.target_bits_per_sample {
                        (s << (self.target_bits_per_sample - self.source_int_bits_per_sample))
                            as f32
                    } else {
                        (s >> (self.source_int_bits_per_sample - self.target_bits_per_sample))
                            as f32
                    }
                })
                .collect::<Vec<f32>>(),
        )
    }
    pub fn reencode_float(&mut self, audio_chunk: &[f32]) -> Vec<f32> {
        self.reencode_to_mono_with_sample_rate(audio_chunk)
            .iter()
            .map(|sample| {
                let sample_value = *sample;
                let cvt_value = if sample_value < 0. {
                    (sample_value * 32768.) as i32 as f32
                } else {
                    (sample_value * 32767.) as i32 as f32
                };
                cvt_value.min(32767.).max(-32768.)
            })
            .collect::<Vec<f32>>()
    }
    fn reencode_to_mono_with_sample_rate(&mut self, buffer: &[f32]) -> Vec<f32> {
        let mono_buffer = if self.source_channels != 1 {
            buffer
                .chunks_exact(self.source_channels as usize)
                .map(|chunk| chunk[0])
                .collect::<Vec<f32>>()
        } else {
            buffer.to_vec()
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
                .process_into_buffer(&waves_in, waves_out, None)
                .unwrap();
            waves_out.get(0).unwrap().to_vec()
        }
    }
    pub fn new(
        input_spec: &WavFmt,
        frame_length_ms: usize,
        target_sample_rate: usize,
        target_bits_per_sample: u16,
    ) -> Result<WAVEncoder, &'static str> {
        let mut input_samples_per_frame =
            input_spec.sample_rate * frame_length_ms / 1000 * input_spec.channels as usize;
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
    audio_buffer: &[u8],
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
    audio_buffer: &[u8],
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
    let encoder = WAVEncoder::new(
        &wav_spec,
        crate::FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        crate::DETECTOR_INTERNAL_SAMPLE_RATE,
        crate::DETECTOR_INTERNAL_BIT_DEPTH,
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
fn reencode_wav_with_different_format_and_rate() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let f32_samples_file =
        std::fs::File::open(dir.to_owned() + "/tests/resources/test_f32.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(f32_samples_file)).unwrap();
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
        crate::FEATURE_EXTRACTOR_FRAME_LENGTH_MS,
        crate::DETECTOR_INTERNAL_SAMPLE_RATE,
        crate::DETECTOR_INTERNAL_BIT_DEPTH,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_ne!(
        input_length, output_length,
        "input and output have same length"
    );
    assert_eq!(input_length, 1440, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
    let samples = wav_reader
        .into_samples::<f32>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let internal_spec = hound::WavSpec {
        sample_rate: crate::DETECTOR_INTERNAL_SAMPLE_RATE as u32,
        bits_per_sample: crate::DETECTOR_INTERNAL_BIT_DEPTH,
        sample_format: hound::SampleFormat::Int,
        channels: 1,
    };
    let mut writer = hound::WavWriter::create(
        dir.to_owned() + "/tests/resources/test_f32_converted_int16.wav",
        internal_spec,
    )
    .unwrap();
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.reencode_float(chuck))
        .for_each(|reencoded_chunk| {
            for sample in reencoded_chunk {
                writer.write_sample(sample as i16).ok();
            }
        });
    writer.finalize().expect("Unable to save file");
}
