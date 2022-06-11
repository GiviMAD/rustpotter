#[cfg(feature = "vad")]
use crate::VadMode;
use crate::{
    detector::INTERNAL_SAMPLE_RATE, Endianness, NoiseDetectionMode, SampleFormat, WakewordDetector,
};

/// Use this struct to configure and build your wakeword detector.
/// ```
/// use rustpotter::{WakewordDetectorBuilder};
/// let mut word_detector = WakewordDetectorBuilder::new()
///     .set_threshold(0.5)
///     .set_sample_rate(16000)
///     .set_eager_mode(true)
///     .set_single_thread(true)
///     .build();
/// ```
pub struct WakewordDetectorBuilder {
    sample_rate: Option<usize>,
    sample_format: Option<SampleFormat>,
    bits_per_sample: Option<u16>,
    channels: Option<u16>,
    endianness: Option<Endianness>,
    eager_mode: bool,
    single_thread: bool,
    averaged_threshold: Option<f32>,
    threshold: Option<f32>,
    comparator_band_size: Option<usize>,
    comparator_ref: Option<f32>,
    noise_mode: Option<NoiseDetectionMode>,
    noise_sensitivity: Option<f32>,
    #[cfg(feature = "vad")]
    vad_mode: Option<VadMode>,
    #[cfg(feature = "vad")]
    vad_sensitivity: Option<f32>,
    #[cfg(feature = "vad")]
    vad_delay: Option<u16>,
}
impl WakewordDetectorBuilder {
    pub fn new() -> Self {
        WakewordDetectorBuilder {
            // input options
            sample_rate: None,
            sample_format: None,
            bits_per_sample: None,
            channels: None,
            endianness: None,
            // detection options
            eager_mode: false,
            single_thread: false,
            threshold: None,
            averaged_threshold: None,
            comparator_band_size: None,
            comparator_ref: None,
            noise_mode: None,
            noise_sensitivity: None,
            #[cfg(feature = "vad")]
            vad_mode: None,
            #[cfg(feature = "vad")]
            vad_delay: None,
            #[cfg(feature = "vad")]
            vad_sensitivity: None,
        }
    }
    /// construct the wakeword detector
    pub fn build(&self) -> WakewordDetector {
        WakewordDetector::new(
            self.get_sample_rate(),
            self.get_sample_format(),
            self.get_bits_per_sample(),
            self.get_channels(),
            self.get_endianness(),
            self.get_eager_mode(),
            self.get_single_thread(),
            self.get_threshold(),
            self.get_averaged_threshold(),
            self.get_comparator_band_size(),
            self.get_comparator_ref(),
            self.get_noise_mode(),
            self.get_noise_sensitivity(),
            #[cfg(feature = "vad")]
            self.get_vad_mode(),
            #[cfg(feature = "vad")]
            self.get_vad_delay(),
            #[cfg(feature = "vad")]
            self.get_vad_sensitivity(),
        )
    }
    /// Configures the detector threshold,
    /// is the min score (in range 0. to 1.) that some of
    /// the wakeword templates should obtain to trigger a detection.
    ///
    /// Defaults to 0.5, wakeword defined value takes prevalence if present.
    pub fn set_threshold(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.threshold = Some(value);
        self
    }
    /// Configures the detector threshold,
    /// is the min score (in range 0. to 1.) that  
    /// the averaged wakeword template should obtain to allow
    /// to continue with the detection. This way it can prevent to
    /// run the comparison of the current frame against each of the wakeword templates.
    /// If set to 0. this functionality is disabled.
    ///
    /// Defaults to half of the configured threshold, wakeword defined value takes prevalence if present.
    pub fn set_averaged_threshold(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.averaged_threshold = Some(value);
        self
    }
    /// Configures the detector expected bit per sample for the audio chunks to process.
    ///
    /// When sample format is set to 'float' this is ignored as only 32 is supported.
    ///
    /// Defaults to 16; Allowed values: 8, 16, 24, 32
    pub fn set_bits_per_sample(&mut self, value: u16) -> &mut Self {
        assert!(
            8 == value || 16 == value || 24 == value || 32 == value,
            "Allowed values are 8, 16, 24 and 32"
        );
        self.bits_per_sample = Some(value);
        self
    }
    /// Configures the detector expected sample rate for the audio chunks to process.
    ///
    /// Defaults to 48000
    pub fn set_sample_rate(&mut self, value: usize) -> &mut Self {
        self.sample_rate = Some(value);
        self
    }
    /// Configures the detector expected sample format for the audio chunks to process.
    ///
    /// Defaults to int
    pub fn set_sample_format(&mut self, value: SampleFormat) -> &mut Self {
        self.sample_format = Some(value);
        self
    }
    /// Configures the detector expected number of channels for the audio chunks to process.
    /// Rustpotter will only use data for first channel.
    ///
    /// Defaults to 1
    pub fn set_channels(&mut self, value: u16) -> &mut Self {
        self.channels = Some(value);
        self
    }
    /// Configures expected endianness for the process_buffer input
    ///
    /// Defaults to little-endian
    pub fn set_endianness(&mut self, value: Endianness) -> &mut Self {
        self.endianness = Some(value);
        self
    }
    /// Configures the band-size for the comparator used to match the samples.
    ///
    /// Defaults to 6
    pub fn set_comparator_band_size(&mut self, value: usize) -> &mut Self {
        self.comparator_band_size = Some(value);
        self
    }
    /// Configures the reference for the comparator used to match the samples.
    ///
    /// Defaults to 0.22
    pub fn set_comparator_ref(&mut self, value: f32) -> &mut Self {
        self.comparator_ref = Some(value);
        self
    }
    /// Enables eager mode.
    /// Terminate the detection as son as one result is above the score,
    /// instead of wait to see if the next frame has a higher score.
    ///
    /// Recommended for real usage.
    ///
    /// Defaults to false.
    pub fn set_eager_mode(&mut self, value: bool) -> &mut Self {
        self.eager_mode = value;
        self
    }
    /// Unless enabled the comparison against multiple wakewords run
    /// in separate threads.
    ///
    /// Defaults to false.
    ///
    /// Only applies when more than a wakeword is loaded.
    pub fn set_single_thread(&mut self, value: bool) -> &mut Self {
        self.single_thread = value;
        self
    }
    /// Noise/silence ratio in the last second to consider noise detected.
    ///
    /// Defaults to 0.5.
    ///
    /// Only applies if noise detection is enabled.
    pub fn set_noise_sensitivity(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.noise_sensitivity = Some(value);
        self
    }
    /// Use build-in noise detection to reduce computation on absence of noise.
    /// Configures how difficult is to considering a frame as noise (the required noise lever)
    /// Unless specified the noise detection is disabled.
    pub fn set_noise_mode(&mut self, value: NoiseDetectionMode) -> &mut Self {
        self.noise_mode = Some(value);
        self
    }
    #[cfg(feature = "vad")]
    /// Seconds to disable the vad detector after voice is detected.
    ///
    /// Defaults to 3.
    ///
    /// Only applies if vad is enabled.
    pub fn set_vad_delay(&mut self, value: u16) -> &mut Self {
        self.vad_delay = Some(value);
        self
    }
    #[cfg(feature = "vad")]
    /// Voice/silence ratio in the last second to consider voice detected.
    ///
    /// Defaults to 0.5.
    ///
    /// Only applies if vad is enabled.
    pub fn set_vad_sensitivity(&mut self, value: f32) -> &mut Self {
        assert!(value >= 0. || value <= 1.);
        self.vad_sensitivity = Some(value);
        self
    }
    #[cfg(feature = "vad")]
    /// Use a vad detector to reduce computation on absence of voice sound.
    ///
    /// Unless specified the vad detector is disabled.
    pub fn set_vad_mode(&mut self, value: VadMode) -> &mut Self {
        self.vad_mode = Some(value);
        self
    }
    fn get_threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5)
    }
    fn get_averaged_threshold(&self) -> f32 {
        self.averaged_threshold.unwrap_or(self.get_threshold() / 2.)
    }
    fn get_sample_rate(&self) -> usize {
        self.sample_rate.unwrap_or(INTERNAL_SAMPLE_RATE)
    }
    fn get_sample_format(&self) -> SampleFormat {
        self.sample_format.unwrap_or(SampleFormat::Int)
    }
    fn get_bits_per_sample(&self) -> u16 {
        if self.get_sample_format() == SampleFormat::Float {
            32
        } else {
            self.bits_per_sample.unwrap_or(16)
        }
    }
    fn get_channels(&self) -> u16 {
        self.channels.unwrap_or(1)
    }
    fn get_endianness(&self) -> Endianness {
        self.endianness.clone().unwrap_or(Endianness::Little)
    }
    fn get_comparator_band_size(&self) -> usize {
        self.comparator_band_size.unwrap_or(6)
    }
    fn get_comparator_ref(&self) -> f32 {
        self.comparator_ref.unwrap_or(0.22)
    }
    fn get_eager_mode(&self) -> bool {
        self.eager_mode
    }
    fn get_single_thread(&self) -> bool {
        self.single_thread
    }
    fn get_noise_mode(&self) -> Option<NoiseDetectionMode> {
        self.noise_mode.clone()
    }
    fn get_noise_sensitivity(&self) -> f32 {
        self.noise_sensitivity.unwrap_or(0.5)
    }
    #[cfg(feature = "vad")]
    fn get_vad_sensitivity(&self) -> f32 {
        self.vad_sensitivity.unwrap_or(0.5)
    }
    #[cfg(feature = "vad")]
    fn get_vad_delay(&self) -> u16 {
        self.vad_delay.unwrap_or(3)
    }
    #[cfg(feature = "vad")]
    fn get_vad_mode(&self) -> Option<VadMode> {
        self.vad_mode.as_ref()?;
        match self.vad_mode.as_ref().unwrap() {
            VadMode::Quality => Some(VadMode::Quality),
            VadMode::LowBitrate => Some(VadMode::LowBitrate),
            VadMode::Aggressive => Some(VadMode::Aggressive),
            VadMode::VeryAggressive => Some(VadMode::VeryAggressive),
        }
    }
}
impl Default for WakewordDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
