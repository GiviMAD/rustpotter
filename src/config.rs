use crate::{
    DETECTOR_DEFAULT_AVG_THRESHOLD, DETECTOR_DEFAULT_THRESHOLD, DETECTOR_INTERNAL_BIT_DEPTH,
    DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
    FEATURE_COMPARATOR_DEFAULT_REFERENCE, DETECTOR_DEFAULT_MIN_SCORES,
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
    /// Indicates the sample rate of the input audio stream.
    pub sample_rate: usize,
    /// Indicates the sample format used to encode the input audio stream bytes.
    pub sample_format: SampleFormat,
    /// Indicates the bit depth used to encode the input audio stream bytes.
    pub bits_per_sample: u16,
    /// Indicates the number of channels of the input audio stream.
    pub channels: u16,
    /// Input the sample endianness used to encode the input audio stream bytes.
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
    /// Enables a gain-normalizer audio filter that normalize the loudness of each input sample buffer 
    /// with respect to the loudness wakeword sample (the RMS level is used as loudness measure).
    pub gain_normalizer: bool,
    /// Enables band-pass audio filter that attenuates frequencies outside that range 
    /// defined by the low_cutoff and high_cutoff values.
    pub band_pass: bool,
    /// Low cutoff for the band-pass filter.
    pub low_cutoff: f32,
    /// High cutoff for the band-pass filter.
    pub high_cutoff: f32,
}
impl Default for AudioFilters {
    fn default() -> AudioFilters {
        AudioFilters {
            gain_normalizer: true,
            band_pass: true,
            low_cutoff: 80.,
            high_cutoff: 400.,
        }
    }
}
pub struct DetectorConfig {
    /// Minimum score against the averaged sample features.
    pub avg_threshold: f32,
    /// Minimum score against one of the sample features.
    pub threshold: f32,
    /// Minimum number of positive scores during detection.
    pub min_scores: usize,
    /// Feature comparator band size.
    pub comparator_band_size: u16,
    /// Feature comparator reference.
    pub comparator_reference: f32,
}
impl Default for DetectorConfig {
    fn default() -> DetectorConfig {
        DetectorConfig {
            avg_threshold: DETECTOR_DEFAULT_AVG_THRESHOLD,
            threshold: DETECTOR_DEFAULT_THRESHOLD,
            min_scores: DETECTOR_DEFAULT_MIN_SCORES,
            comparator_band_size: FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
            comparator_reference: FEATURE_COMPARATOR_DEFAULT_REFERENCE,
        }
    }
}
pub struct RustpotterConfig {
    /// configures expected wav input format.
    pub fmt: WavFmt,
    /// Configures detection.
    pub detector: DetectorConfig,
    /// Configures input audio filters.
    pub filters: AudioFilters,
}
impl Default for RustpotterConfig {
    fn default() -> RustpotterConfig {
        RustpotterConfig {
            fmt: WavFmt::default(),
            detector: DetectorConfig::default(),
            filters: AudioFilters::default(),
        }
    }
}