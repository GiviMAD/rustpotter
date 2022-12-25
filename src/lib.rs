/// Rustpotter, an open source wake word spotter forged in rust.
extern crate savefile;
#[macro_use]
extern crate savefile_derive;

mod comparator;
mod utils;
mod detector;
mod detector_builder;
mod dtw;
mod wakeword;
pub use detector::DetectedWakeword;
pub use detector::NoiseDetectionMode;
pub use detector::SampleFormat;
pub use detector::Endianness;
#[cfg(feature = "vad")]
pub use detector::VadMode;
pub use detector::WakewordDetector;
pub use detector_builder::WakewordDetectorBuilder;
