use crate::constants::{
    DETECTOR_DEFAULT_AVG_THRESHOLD, DETECTOR_DEFAULT_MIN_SCORES, DETECTOR_DEFAULT_THRESHOLD,
    DETECTOR_INTERNAL_SAMPLE_RATE, FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
    FEATURE_COMPARATOR_DEFAULT_REFERENCE,
};

/// Indicates the byte endianness
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Clone, Copy)]
pub enum Endianness {
    Big,
    Little,
    Native,
}
/// Supported sample formats
pub type SampleFormat = hound::SampleFormat;

/// Wav format representation
#[cfg_attr(feature = "debug", derive(Debug))]
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
            sample_format: hound::SampleFormat::Float,
            bits_per_sample: 32,
            channels: 1,
            endianness: Endianness::Little,
        }
    }
}
/// Configures the gain-normalizer audio filter used.
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct GainNormalizationConfig {
    /// Enables the filter.
    pub enabled: bool,
    /// Set the rms level reference used to calculate the gain applied.
    /// If unset the estimated wakeword rms level is used.
    pub gain_ref: Option<f32>,
    /// Min gain applied. (precision of 0.1)
    pub min_gain: f32,
    /// Max gain applied. (precision of 0.1)
    pub max_gain: f32,
}
impl Default for GainNormalizationConfig {
    fn default() -> GainNormalizationConfig {
        GainNormalizationConfig {
            enabled: false,
            gain_ref: None,
            min_gain: 0.1,
            max_gain: 1.0,
        }
    }
}
/// Configures the band-pass audio filter used.
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct BandPassConfig {
    /// Enables the filter.
    pub enabled: bool,
    /// Low cutoff for the band-pass filter.
    pub low_cutoff: f32,
    /// High cutoff for the band-pass filter.
    pub high_cutoff: f32,
}
impl Default for BandPassConfig {
    fn default() -> BandPassConfig {
        BandPassConfig {
            enabled: false,
            low_cutoff: 80.,
            high_cutoff: 400.,
        }
    }
}
/// Configures the audio filters.
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Default)]
pub struct FiltersConfig {
    /// Enables a gain-normalizer audio filter that intent to approximate the volume of the stream
    /// to a reference level (RMS of the samples is used as volume measure).
    pub gain_normalizer: GainNormalizationConfig,
    /// Enables a band-pass audio filter that attenuates frequencies outside the low cutoff and high cutoff range.
    pub band_pass: BandPassConfig,
}

/// Indicates how to calculate the final score.
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Clone, Copy)]
pub enum ScoreMode {
    Average,
    Max,
    Median,
    P25,
    P50,
    P75,
    P80,
    P90,
    P95,
}
/// Configures the detector scoring behavior.
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct DetectorConfig {
    /// Minimum required score against the wakeword averaged feature frame vector.
    pub avg_threshold: f32,
    /// Minimum required score against the some of the wakeword feature frame vectors.
    pub threshold: f32,
    /// Minimum number of positive scores during detection.
    pub min_scores: usize,
    /// How to calculate a unified score.
    pub score_mode: ScoreMode,
    /// Feature comparator band size.
    pub comparator_band_size: u16,
    /// Feature comparator reference. (Used to express the similarity as a percent)
    pub comparator_ref: f32,
}
impl Default for DetectorConfig {
    fn default() -> DetectorConfig {
        DetectorConfig {
            avg_threshold: DETECTOR_DEFAULT_AVG_THRESHOLD,
            threshold: DETECTOR_DEFAULT_THRESHOLD,
            min_scores: DETECTOR_DEFAULT_MIN_SCORES,
            score_mode: ScoreMode::Max,
            comparator_band_size: FEATURE_COMPARATOR_DEFAULT_BAND_SIZE,
            comparator_ref: FEATURE_COMPARATOR_DEFAULT_REFERENCE,
        }
    }
}
/// Encapsulates all the tool configurations.
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Default)]
pub struct RustpotterConfig {
    /// configures expected wav input format.
    pub fmt: WavFmt,
    /// Configures detection.
    pub detector: DetectorConfig,
    /// Configures input audio filters.
    pub filters: FiltersConfig,
}
