mod wakeword_model_train;
mod wakeword_nn;

pub use wakeword_model_train::{WakewordModelTrain, WakewordModelTrainConfig};
pub(crate) use wakeword_nn::WakewordNN;
