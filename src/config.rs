use crate::{
    DETECTOR_DEFAULT_AVG_THRESHOLD, DETECTOR_DEFAULT_THRESHOLD, DETECTOR_INTERNAL_BIT_DEPTH,
    DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
    FEATURE_COMPARATOR_DEFAULT_REFERENCE,
};

/// Supported endianness modes
#[derive(Clone)]
pub enum Endianness {
    Big,
    Little,
    Native,
}
/// Supported sample formats
pub type SampleFormat = hound::SampleFormat;

/// Wav spec representation
pub struct WavFmt {
    pub sample_rate: usize,
    pub sample_format: SampleFormat,
    pub bits_per_sample: u16,
    pub channels: u16,
    pub endianness: Endianness,
}
impl Default for WavFmt {
    fn default() -> WavFmt {
        WavFmt {
            sample_rate: DETECTOR_INTERNAL_SAMPLE_RATE,
            sample_format: hound::SampleFormat::Int,
            bits_per_sample: DETECTOR_INTERNAL_BIT_DEPTH,
            channels: 1,
            endianness: Endianness::Little,
        }
    }
}
pub struct RustpotterConfig {
    // Wav audio format expected by the detector
    pub fmt: WavFmt,
    pub avg_threshold: f32,
    pub threshold: f32,
    pub comparator_band_size: u16,
    pub comparator_reference: f32,
}
impl Default for RustpotterConfig {
    fn default() -> RustpotterConfig {
        RustpotterConfig {
            fmt: WavFmt::default(),
            avg_threshold: DETECTOR_DEFAULT_AVG_THRESHOLD,
            threshold: DETECTOR_DEFAULT_THRESHOLD,
            comparator_band_size: FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
            comparator_reference: FEATURE_COMPARATOR_DEFAULT_REFERENCE,
        }
    }
}
