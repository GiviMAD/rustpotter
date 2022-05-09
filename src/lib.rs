/// Rustpotter, a free and open source wake word spotter forged in rust.

extern crate simple_matrix;
extern crate savefile;
#[macro_use]
extern crate savefile_derive;

mod comparator;
mod detector;
mod dtw;
mod nnnoiseless_fork;
mod wakeword;
pub use detector::DetectedWakeword;
pub use detector::SampleFormat;
pub use detector::VadMode;
pub use detector::WakewordDetector;
pub use detector::WakewordDetectorBuilder;
