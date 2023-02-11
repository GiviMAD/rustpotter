use crate::{
    DETECTOR_DEFAULT_AVG_THRESHOLD, DETECTOR_DEFAULT_THRESHOLD, DETECTOR_INTERNAL_BIT_DEPTH,
    DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
    FEATURE_COMPARATOR_DEFAULT_REFERENCE,
};

/// Indicates the byte endianness
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
    /// Indicates the sample rate of the input
    pub sample_rate: usize,
    /// Indicates the sample format of the input bytes
    pub sample_format: SampleFormat,
    /// Indicates the bit depth of the input samples
    pub bits_per_sample: u16,
    /// Indicates the number of channels in the input
    pub channels: u16,
    /// Input the sample endianness of the input bytes
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
pub struct AudioFilters {
    /// Enables band-pass audio filter
    pub band_pass: bool,
    /// Low cutoff for the band-pass filter
    pub low_cutoff: f32,
    /// High cutoff for the band-pass filter
    pub high_cutoff: f32,
}
impl Default for AudioFilters {
    fn default() -> AudioFilters {
        AudioFilters {
            band_pass: true,
            low_cutoff: 80.,
            high_cutoff: 400.,
        }
    }
}
pub struct DetectorConfig {
    /// Minimum score against the averaged sample features
    pub avg_threshold: f32,
    /// Minimum score against one of the sample features
    pub threshold: f32,
    /// Feature comparator band size
    pub comparator_band_size: u16,
    /// Feature comparator reference
    pub comparator_reference: f32,
}
impl Default for DetectorConfig {
    fn default() -> DetectorConfig {
        DetectorConfig {
            avg_threshold: DETECTOR_DEFAULT_AVG_THRESHOLD,
            threshold: DETECTOR_DEFAULT_THRESHOLD,
            comparator_band_size: FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
            comparator_reference: FEATURE_COMPARATOR_DEFAULT_REFERENCE,
        }
    }
}
pub struct RustpotterConfig {
    /// configures expected wav input format
    pub fmt: WavFmt,
    /// Configures detection
    pub detector: DetectorConfig,
    /// Configures input audio filters
    pub filters: AudioFilters,
}
impl Default for RustpotterConfig {
    fn default() -> RustpotterConfig {
        RustpotterConfig {
            fmt: WavFmt::default(),
            detector: DetectorConfig::default(),
            filters: AudioFilters::default()
        }
    }
}
