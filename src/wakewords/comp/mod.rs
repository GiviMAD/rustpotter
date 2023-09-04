mod wakeword_comp;
mod wakeword_ref_build;
pub(crate) use wakeword_comp::WakewordComparator;
pub use wakeword_ref_build::{WakewordRefBuildFromBuffers, WakewordRefBuildFromFiles};
