mod wakeword_detector;
mod wakeword_model;
mod wakeword_ref;
mod wakeword_serde;

pub(crate) use wakeword_detector::WakewordDetector;
pub use wakeword_ref::WakewordRef;
pub use wakeword_model::{ModelWeights, TensorData, WakewordModel};
pub use wakeword_serde::{DeserializableWakeword, SerializableWakeword};
