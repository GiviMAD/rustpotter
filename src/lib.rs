mod audio;
mod config;
mod constants;
mod detector;
mod mfcc;
mod wakewords;
#[cfg(feature = "audio")]
pub use audio::{BandPassFilter, GainNormalizerFilter, WAVEncoder};
pub use audio::{Endianness, Sample, SampleFormat};
pub use config::{
    AudioFmt, BandPassConfig, DetectorConfig, FiltersConfig, GainNormalizationConfig,
    RustpotterConfig, ScoreMode, VADMode,
};
#[cfg(feature = "audio")]
pub use constants::{DETECTOR_INTERNAL_SAMPLE_RATE, MFCCS_EXTRACTOR_FRAME_LENGTH_MS};
pub use detector::{Rustpotter, RustpotterDetection};
pub use wakewords::{
    ModelType, ModelWeights, TensorData, WakewordLoad, WakewordModel, WakewordModelTrain,
    WakewordModelTrainOptions, WakewordRef, WakewordRefBuildFromBuffers, WakewordRefBuildFromFiles,
    WakewordSave,
};
