use crate::Endianness;
use rubato::Resampler;
pub fn resample_audio_chunk(resampler: &mut rubato::FftFixedInOut<f32>, resampler_out_buffer: &mut [Vec<f32>], float_buffer: Vec<f32>) -> Vec<f32> {
    let waves_in = vec![float_buffer; 1];
    let waves_out = resampler_out_buffer;
    resampler
        .process_into_buffer(&waves_in, waves_out, None)
        .unwrap();
    waves_out.get(0).unwrap().to_vec()
}
pub fn to_pcm_16_bit_per_sample(audio_data: Vec<i32>, bits_per_sample: u16) -> Vec<f32> {
    audio_data.iter()
            .map(|s| {
                if bits_per_sample < 16 {
                    (s << (16 - bits_per_sample)) as f32
                } else {
                    (s >> (bits_per_sample - 16)) as f32
                }
            })
            .collect::<Vec<f32>>()
}
pub fn encode_float_audio_to_pcm_16_bit_mono(audio_chunk: Vec<f32>, channels: u16) -> Vec<f32>{
    if channels != 1 {
        audio_chunk
            .chunks_exact(channels as usize)
            .map(|chunk| chunk[0])
            .map(convert_f32_sample)
            .collect::<Vec<f32>>()
    } else {
        audio_chunk
            .iter()
            .map(|sample|convert_f32_sample(*sample))
            .collect::<Vec<f32>>()
    }
}

pub fn decode_int_audio_bytes(audio_buffer: Vec<u8>, bits_per_sample: u16, endianness: &Endianness) -> Vec<i32> {
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


pub fn decode_float_audio_bytes(audio_buffer: Vec<u8>, bits_per_sample: u16, endianness: &Endianness) -> Vec<f32> {
    let buffer_chunks = audio_buffer.chunks_exact((bits_per_sample / 8) as usize);
    let audio_chunk: Vec<f32> = match endianness {
        Endianness::Little => 
        buffer_chunks
        .map(|bytes| {
            match bits_per_sample {
                32 => {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                }
                _default => {
                    0.
                }
            }
        }).collect::<Vec<f32>>(),
        Endianness::Big => 
        buffer_chunks
        .map(|bytes| {
            match bits_per_sample {
                32 => {
                    f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                }
                _default => {
                    0.
                }
            }
        }).collect::<Vec<f32>>(),
        Endianness::Native => 
        buffer_chunks
        .map(|bytes| {
            match bits_per_sample {
                32 => {
                    f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                }
                _default => {
                    0.
                }
            }
        }).collect::<Vec<f32>>()
    };
    audio_chunk
}

pub fn convert_f32_sample(value: f32) -> f32 {
    if value < 0. { value * 0x8000 as f32} else {value * 0x7fff as f32}
}