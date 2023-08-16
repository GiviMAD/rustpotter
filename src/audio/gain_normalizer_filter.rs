pub struct GainNormalizerFilter {
    window_size: usize,
    fixed_rms_level: bool,
    min_gain: f32,
    max_gain: f32,
    // state
    rms_level_ref: f32,
    rms_level_sqrt: f32,
    rms_level_window: Vec<f32>,
}
impl GainNormalizerFilter {
    pub fn filter(&mut self, signal: &mut [f32], rms_level: f32) -> f32 {
        if !self.rms_level_ref.is_nan() && rms_level != 0. {
            // update the window
            self.rms_level_window.push(rms_level);
            if self.rms_level_window.len() > self.window_size {
                self.rms_level_window.drain(0..1);
            }
            // calculate the mean of the rms window
            let frame_rms_level =
                self.rms_level_window.iter().sum::<f32>() / self.rms_level_window.len() as f32;
            // calculate the gain to apply
            let mut gain = self.rms_level_sqrt / frame_rms_level.sqrt();
            // range and round the value, trying to get a mostly uniform gain between frames.
            gain = ((gain * 10.).round() / 10.).clamp(self.min_gain, self.max_gain);
            // apply gain unless irrelevant
            if gain != 1. {
                for sample in signal {
                    *sample = (*sample * gain).clamp(-1., 1.);
                }
            }
            gain
        } else {
            1.
        }
    }
    pub fn get_rms_level_ref(&self) -> f32 {
        self.rms_level_ref
    }
    pub fn set_rms_level_ref(&mut self, rms_level: f32, window_size: usize) {
        if !self.fixed_rms_level {
            self.rms_level_ref = rms_level;
            self.rms_level_sqrt = rms_level.sqrt();
        }
        self.window_size = if window_size != 0 { window_size } else { 1 };
    }
    pub fn get_rms_level(signal: &[f32]) -> f32 {
        let mut sum_squared = 0.0;
        for sample in signal {
            sum_squared += sample * sample;
        }
        (sum_squared / signal.len() as f32).sqrt()
    }
    pub fn new(min_gain: f32, max_gain: f32, fixed_rms_level: Option<f32>) -> GainNormalizerFilter {
        GainNormalizerFilter {
            min_gain,
            max_gain,
            rms_level_ref: fixed_rms_level.unwrap_or(f32::NAN),
            rms_level_sqrt: fixed_rms_level.map(|s| s.sqrt()).unwrap_or(f32::NAN),
            fixed_rms_level: fixed_rms_level.is_some(),
            rms_level_window: Vec::new(),
            window_size: 1,
        }
    }
}

#[test]
fn filter_audio() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_file =
        std::fs::File::open(dir.to_owned() + "/tests/resources/real_sample.wav").unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(sample_file)).unwrap();
    let wav_spec = crate::WavFmt {
        sample_rate: wav_reader.spec().sample_rate as usize,
        sample_format: wav_reader.spec().sample_format,
        bits_per_sample: wav_reader.spec().bits_per_sample,
        channels: wav_reader.spec().channels,
        endianness: crate::Endianness::Little,
    };
    println!("{:?}", wav_spec);
    let mut encoder = crate::audio::WAVEncoder::new(
        &wav_spec,
        crate::constants::MFCCS_EXTRACTOR_FRAME_LENGTH_MS,
        crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE,
    )
    .unwrap();
    let input_length = encoder.get_input_frame_length();
    let output_length = encoder.get_output_frame_length();
    assert_ne!(
        input_length, output_length,
        "input and output not have same length"
    );
    assert_eq!(input_length, 1440, "input size is correct");
    assert_eq!(output_length, 480, "output size is correct");
    let samples = wav_reader
        .into_samples::<f32>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let internal_spec = hound::WavSpec {
        sample_rate: crate::constants::DETECTOR_INTERNAL_SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        channels: 1,
    };
    let mut writer = hound::WavWriter::create(
        dir.to_owned() + "/tests/resources/gain-normalizer_example.wav",
        internal_spec,
    )
    .unwrap();
    let mut filter = GainNormalizerFilter::new(0.1, 1., Some(0.003));
    samples
        .chunks_exact(encoder.get_input_frame_length())
        .map(|chuck| encoder.rencode_and_resample::<f32>(chuck.into()))
        .map(|mut chunk| {
            let rms_level = GainNormalizerFilter::get_rms_level(&chunk);
            filter.filter(&mut chunk, rms_level);
            chunk
        })
        .for_each(|encoded_chunk| {
            for sample in encoded_chunk {
                writer.write_sample(sample).ok();
            }
        });
    writer.finalize().expect("Unable to save file");
}
