/// Indicates the sample format type and size
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Clone)]
pub enum SampleFormat {
    I8,
    I16,
    I32,
    F32,
}
impl SampleFormat {
    pub fn get_bits_per_sample(&self) -> u16 {
        match self {
            SampleFormat::I8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I32 => 32,
            SampleFormat::F32 => 32,
        }
    }
    pub fn get_bytes_per_sample(&self) -> u16 {
        self.get_bits_per_sample() / 8
    }
    pub fn int_of_size(bit_size: u16) -> Option<Self> {
        match bit_size {
            8 => Some(SampleFormat::I8),
            16 => Some(SampleFormat::I16),
            32 => Some(SampleFormat::I32),
            _ => None,
        }
    }
    pub fn float_of_size(bit_size: u16) -> Option<Self> {
        match bit_size {
            32 => Some(SampleFormat::F32),
            _ => None,
        }
    }
}
#[cfg(feature = "display")]
impl std::fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SampleFormat::I8 => write!(f, "i8"),
            SampleFormat::I16 => write!(f, "i16"),
            SampleFormat::I32 => write!(f, "i32"),
            SampleFormat::F32 => write!(f, "f32"),
        }
    }
}

/// Indicates the sample byte order in the audio byte stream
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Clone)]
pub enum Endianness {
    Big,
    Little,
    Native,
}

/// Trait for compatible sample number types
pub trait Sample: Sized + Copy + std::cmp::PartialOrd + 'static + Send {
    const S_TYPE: SampleFormat;
    fn get_byte_size() -> usize;
    fn get_format() -> SampleFormat;
    fn get_zero() -> Self;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn from_be_bytes(bytes: &[u8]) -> Self;
    fn from_ne_bytes(bytes: &[u8]) -> Self;
    fn into_f32(self) -> f32;
}

macro_rules! with_sample_type {
    ($ty:ty, $dsample:ident, $byte_size:literal, $to_f32:expr, $from_le_bytes:expr, $from_be_bytes:expr, $from_ne_bytes:expr, $zero:literal) => {
        impl Sample for $ty {
            const S_TYPE: SampleFormat = SampleFormat::$dsample;
            fn get_byte_size() -> usize {
                $byte_size
            }
            fn get_format() -> SampleFormat {
                Self::S_TYPE
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
    I8,
    1,
    |v: i8| v as f32 / (i8::MAX as f32),
    |bytes: &[u8]| i8::from_le_bytes([bytes[0]]),
    |bytes: &[u8]| i8::from_be_bytes([bytes[0]]),
    |bytes: &[u8]| i8::from_ne_bytes([bytes[0]]),
    0
);
with_sample_type!(
    i16,
    I16,
    2,
    |v: i16| v as f32 / (i16::MAX as f32),
    |bytes: &[u8]| i16::from_le_bytes([bytes[0], bytes[1]]),
    |bytes: &[u8]| i16::from_be_bytes([bytes[0], bytes[1]]),
    |bytes: &[u8]| i16::from_ne_bytes([bytes[0], bytes[1]]),
    0
);
with_sample_type!(
    i32,
    I32,
    4,
    |v: i32| v as f32 / (i32::MAX as f32),
    |bytes: &[u8]| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    0
);
with_sample_type!(
    f32,
    F32,
    4,
    |v: f32| v,
    |bytes: &[u8]| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    |bytes: &[u8]| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    0.
);
