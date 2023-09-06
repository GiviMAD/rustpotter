use crate::{
    audio::{Endianness, SampleFormat},
    constants::{
        COMPARATOR_DEFAULT_BAND_SIZE, DETECTOR_DEFAULT_AVG_THRESHOLD, DETECTOR_DEFAULT_MIN_SCORES,
        DETECTOR_DEFAULT_REFERENCE, DETECTOR_DEFAULT_THRESHOLD, DETECTOR_INTERNAL_SAMPLE_RATE,
    },
};
/// Wav format representation
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct AudioFmt {
    /// Indicates the sample rate of the input audio stream.
    pub sample_rate: usize,
    /// Indicates the sample type and its bit size. It's only used when the audio is provided as bytes.
    pub sample_format: SampleFormat,
    /// Indicates the number of channels of the input audio stream.
    pub channels: u16,
    /// Input the sample endianness used to encode the input audio stream bytes.
    pub endianness: Endianness,
}
impl Default for AudioFmt {
    fn default() -> AudioFmt {
        AudioFmt {
            sample_rate: DETECTOR_INTERNAL_SAMPLE_RATE,
            sample_format: SampleFormat::F32,
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
#[cfg(feature = "display")]
impl std::fmt::Display for ScoreMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ScoreMode::Average => write!(f, "average"),
            ScoreMode::Max => write!(f, "max"),
            ScoreMode::Median => write!(f, "median"),
            ScoreMode::P25 => write!(f, "p25"),
            ScoreMode::P50 => write!(f, "p50"),
            ScoreMode::P75 => write!(f, "p75"),
            ScoreMode::P80 => write!(f, "p80"),
            ScoreMode::P90 => write!(f, "p90"),
            ScoreMode::P95 => write!(f, "p95"),
        }
    }
}
#[cfg(feature = "display")]
impl std::str::FromStr for ScoreMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "average" => Ok(Self::Average),
            "max" => Ok(Self::Max),
            "median" => Ok(Self::Median),
            "p25" => Ok(Self::P25),
            "p50" => Ok(Self::P50),
            "p75" => Ok(Self::P75),
            "p80" => Ok(Self::P80),
            "p90" => Ok(Self::P90),
            "p95" => Ok(Self::P95),
            _ => Err("Unknown score mode".to_string()),
        }
    }
}
/// Configures VAD detector sensibility.
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Clone, Copy)]
pub enum VADMode {
    Easy,
    Medium,
    Hard,
}
impl VADMode {
    pub(crate) fn get_value(&self) -> f32 {
        match &self {
            VADMode::Easy => 2.,
            VADMode::Medium => 2.5,
            VADMode::Hard => 3.,
        }
    }
}
#[cfg(feature = "display")]
impl std::fmt::Display for VADMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            VADMode::Easy => write!(f, "easy"),
            VADMode::Medium => write!(f, "medium"),
            VADMode::Hard => write!(f, "hard"),
        }
    }
}
#[cfg(feature = "display")]
impl std::str::FromStr for VADMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "easy" => Ok(Self::Easy),
            "medium" => Ok(Self::Medium),
            "hard" => Ok(Self::Hard),
            _ => Err("Unknown vad mode".to_string()),
        }
    }
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
    /// Emit detection on min partial scores.
    pub eager: bool,
    /// Value used to express the score as a percent in range 0 - 1.
    pub score_ref: f32,
    /// Comparator band size. Doesn't apply to wakeword models.
    pub band_size: u16,
    /// How to calculate a unified score. Doesn't apply to wakeword models.
    pub score_mode: ScoreMode,
    /// How to calculate a unified score. Doesn't apply to wakeword models.
    pub vad_mode: Option<VADMode>,
    #[cfg(feature = "record")]
    /// Path to create records, one on the first partial detection and another each one that scores better.
    pub record_path: Option<String>,
}
impl Default for DetectorConfig {
    fn default() -> DetectorConfig {
        DetectorConfig {
            avg_threshold: DETECTOR_DEFAULT_AVG_THRESHOLD,
            threshold: DETECTOR_DEFAULT_THRESHOLD,
            min_scores: DETECTOR_DEFAULT_MIN_SCORES,
            score_ref: DETECTOR_DEFAULT_REFERENCE,
            band_size: COMPARATOR_DEFAULT_BAND_SIZE,
            vad_mode: None,
            score_mode: ScoreMode::Max,
            eager: false,
            #[cfg(feature = "record")]
            record_path: None,
        }
    }
}
/// Encapsulates all the tool configurations.
#[cfg_attr(feature = "debug", derive(Debug))]
#[derive(Default)]
pub struct RustpotterConfig {
    /// Configures expected audio input format.
    pub fmt: AudioFmt,
    /// Configures detection.
    pub detector: DetectorConfig,
    /// Configures input audio filters.
    pub filters: FiltersConfig,
}
