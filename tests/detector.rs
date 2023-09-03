use std::{
    fs::File,
    io::{BufReader, Read},
};

use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat, ScoreMode, VADMode};

#[test]
fn it_can_detect_wakewords_with_v2_file() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g_v2.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].avg_score, 0.6495044);
    assert_eq!(detected_wakewords[0].score, 0.7310586);
    assert_eq!(detected_wakewords[1].avg_score, 0.5804737);
    assert_eq!(detected_wakewords[1].score, 0.721843);
}

#[test]
fn it_can_detect_wakewords_with_max_score_mode() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].avg_score, 0.6495044);
    assert_eq!(detected_wakewords[0].score, 0.7310586);
    assert_eq!(detected_wakewords[1].avg_score, 0.5804737);
    assert_eq!(detected_wakewords[1].score, 0.721843);
}

#[test]
fn it_can_detect_wakewords_with_median_score_mode() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Median;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].avg_score, 0.64608675);
    assert_eq!(detected_wakewords[0].score, 0.60123634);
    assert_eq!(detected_wakewords[1].avg_score, 0.5288923);
    assert_eq!(detected_wakewords[1].score, 0.63968724);
}

#[test]
fn it_can_detect_wakewords_with_average_score_mode() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Average;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].avg_score, 0.64608675);
    assert_eq!(detected_wakewords[0].score, 0.60458726);
    assert_eq!(detected_wakewords[1].avg_score, 0.5750509);
    assert_eq!(detected_wakewords[1].score, 0.6313083);
}

#[test]
fn it_can_detect_wakewords_with_vad_mode() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    config.detector.vad_mode = Some(VADMode::Easy);
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].avg_score, 0.6495044);
    assert_eq!(detected_wakewords[0].score, 0.7310586);
    assert_eq!(detected_wakewords[1].avg_score, 0.5804737);
    assert_eq!(detected_wakewords[1].score, 0.721843);
}

#[test]
fn it_can_ignore_words() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.45;
    config.detector.min_scores = 0;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/alexa.rpw");
    assert_eq!(detected_wakewords.len(), 0);
}
#[test]
fn it_can_ignore_words_while_applying_audio_filters() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.45;
    config.detector.min_scores = 0;
    config.filters.gain_normalizer.enabled = true;
    config.filters.band_pass.enabled = true;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/alexa.rpw");
    assert_eq!(detected_wakewords.len(), 0);
}
#[test]
fn it_can_detect_wakewords_while_applying_band_pass_audio_filter() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = true;
    config.filters.band_pass.low_cutoff = 80.0;
    config.filters.band_pass.high_cutoff = 400.0;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords = run_detection_simulation(config, "/tests/resources/oye_casa_g.rpw");
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].score, 0.6858197);
    assert_eq!(detected_wakewords[1].score, 0.66327363);
}

#[test]
fn it_can_detect_wakewords_while_applying_gain_normalizer_audio_filter() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = true;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    let detected_wakewords =
        run_detection_simulation_with_gains(config, "/tests/resources/oye_casa_g.rpw", 0.2, 5.);
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].score, 0.7304294);
    assert_eq!(detected_wakewords[1].score, 0.71067876);
}

#[test]
fn it_can_detect_wakewords_while_applying_gain_normalizer_and_band_pass_audio_filters() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer.enabled = true;
    config.filters.band_pass.enabled = true;
    config.filters.band_pass.low_cutoff = 80.0;
    config.filters.band_pass.high_cutoff = 500.0;
    config.detector.score_mode = ScoreMode::Median;
    let detected_wakewords =
        run_detection_simulation_with_gains(config, "/tests/resources/oye_casa_g.rpw", 0.2, 5.);
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].score, 0.5775406);
    assert_eq!(detected_wakewords[1].score, 0.5828697);
}

#[test]
fn it_can_detect_wakewords_on_record_with_noise() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.3;
    config.detector.threshold = 0.47;
    config.filters.gain_normalizer.enabled = false;
    config.filters.band_pass.enabled = false;
    config.detector.score_mode = ScoreMode::Max;
    config.detector.min_scores = 5;
    let detected_wakewords = run_detection_with_audio_file(
        config,
        "/tests/resources/oye_casa_real.rpw",
        "/tests/resources/real_sample.wav",
    );
    assert_eq!(detected_wakewords.len(), 3);
    assert_eq!(detected_wakewords[0].avg_score, 0.4676845);
    assert_eq!(detected_wakewords[0].score, 0.527971);
    assert_eq!(detected_wakewords[0].counter, 24);
    assert_eq!(detected_wakewords[1].avg_score, 0.32865646);
    assert_eq!(detected_wakewords[1].score, 0.48120698);
    assert_eq!(detected_wakewords[1].counter, 7);
    assert_eq!(detected_wakewords[2].avg_score, 0.30807483);
    assert_eq!(detected_wakewords[2].score, 0.5164661);
    assert_eq!(detected_wakewords[2].counter, 35);
}

#[test]
fn it_can_detect_wakewords_on_record_with_noise_using_filters() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.3;
    config.detector.threshold = 0.49;
    config.filters.gain_normalizer.enabled = true;
    config.filters.gain_normalizer.min_gain = 0.4;
    config.filters.band_pass.enabled = true;
    config.filters.band_pass.low_cutoff = 210.0;
    config.filters.band_pass.high_cutoff = 700.0;
    config.detector.score_mode = ScoreMode::Max;
    config.detector.min_scores = 5;
    let detected_wakewords = run_detection_with_audio_file(
        config,
        "/tests/resources/oye_casa_real.rpw",
        "/tests/resources/real_sample.wav",
    );
    assert_eq!(detected_wakewords.len(), 3);
    assert_eq!(detected_wakewords[0].avg_score, 0.45496628);
    assert_eq!(detected_wakewords[0].score, 0.5380342);
    assert_eq!(detected_wakewords[0].counter, 23);
    assert_eq!(detected_wakewords[1].avg_score, 0.336222);
    assert_eq!(detected_wakewords[1].score, 0.5001262);
    assert_eq!(detected_wakewords[1].counter, 5);
    assert_eq!(detected_wakewords[2].avg_score, 0.3049497);
    assert_eq!(detected_wakewords[2].score, 0.5189481);
    assert_eq!(detected_wakewords[2].counter, 31);
}

#[test]
fn it_can_detect_wakewords_using_trained_model() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    let detected_wakewords = run_detection_with_audio_file(
        config,
        "/tests/resources/ok_casa-tiny.rpw",
        "/tests/resources/ok_casa.wav",
    );
    assert_eq!(detected_wakewords.len(), 1);
    assert_eq!(detected_wakewords[0].counter, 34);
    assert_eq!(detected_wakewords[0].avg_score, 0.);
    assert_eq!(detected_wakewords[0].score, 0.9997649);
    assert_eq!(detected_wakewords[0].scores["ok_casa"], 3.7506533);
    assert_eq!(detected_wakewords[0].scores["none"], -16.83091);
}

#[test]
fn it_can_detect_wakewords_using_trained_model_and_avg_score() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.5;
    let detected_wakewords = run_detection_with_audio_file(
        config,
        "/tests/resources/ok_casa-tiny.rpw",
        "/tests/resources/ok_casa.wav",
    );
    assert_eq!(detected_wakewords.len(), 1);
    assert_eq!(detected_wakewords[0].counter, 34);
    assert_eq!(detected_wakewords[0].avg_score, 0.9997649);
    assert_eq!(detected_wakewords[0].score, 0.9997649);
    assert_eq!(detected_wakewords[0].scores["ok_casa"], 3.7506533);
    assert_eq!(detected_wakewords[0].scores["none"], -16.83091);
}

#[test]
fn it_can_detect_wakewords_in_eager_mode() {
    let mut config = RustpotterConfig::default();
    config.detector.avg_threshold = 0.;
    config.detector.min_scores = 20;
    config.detector.eager = true;
    let detected_wakewords = run_detection_with_audio_file(
        config,
        "/tests/resources/ok_casa-tiny.rpw",
        "/tests/resources/ok_casa.wav",
    );
    assert_eq!(detected_wakewords.len(), 1);
    assert_eq!(detected_wakewords[0].counter, 20);
    assert_eq!(detected_wakewords[0].avg_score, 0.);
    assert_eq!(detected_wakewords[0].score, 0.9992142);
    assert_eq!(detected_wakewords[0].scores["ok_casa"], 23.990948);
    assert_eq!(detected_wakewords[0].scores["none"], 6.0654087);
}
#[test]
fn it_can_remove_wakeword_by_key() {
    let config = RustpotterConfig::default();
    let mut detector = Rustpotter::new(&config).unwrap();
    let wakeword_key = "test_key";
    let dir = env!("CARGO_MANIFEST_DIR");
    let wakeword_path = dir.to_owned() + "/tests/resources/ok_casa-tiny.rpw";
    detector
        .add_wakeword_from_file(wakeword_key, &wakeword_path)
        .unwrap();
    let result = detector.remove_wakeword(wakeword_key);
    assert!(result, "Wakeword removed");
}

#[test]
fn it_can_remove_all_wakewords() {
    let config = RustpotterConfig::default();
    let mut detector = Rustpotter::new(&config).unwrap();
    let wakeword_key = "test_key";
    let dir = env!("CARGO_MANIFEST_DIR");
    let wakeword_path = dir.to_owned() + "/tests/resources/ok_casa-tiny.rpw";
    detector
        .add_wakeword_from_file(wakeword_key, &wakeword_path)
        .unwrap();
    let result = detector.remove_wakewords();
    assert!(result, "Wakewords removed");
}

fn run_detection_with_audio_file(
    mut config: RustpotterConfig,
    model_path: &str,
    audio_path: &str,
) -> Vec<rustpotter::RustpotterDetection> {
    let dir = env!("CARGO_MANIFEST_DIR");
    let audio_file = std::fs::File::open(dir.to_owned() + audio_path).unwrap();
    let wav_reader = hound::WavReader::new(std::io::BufReader::new(audio_file)).unwrap();
    let wav_spec: rustpotter::AudioFmt = wav_reader.spec().try_into().unwrap();
    config.fmt = wav_spec;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + model_path;
    rustpotter
        .add_wakeword_from_file("wakeword", &model_path)
        .unwrap();
    let mut audio_samples = wav_reader
        .into_samples::<f32>()
        .map(|chunk| *chunk.as_ref().unwrap())
        .collect::<Vec<_>>();
    let mut silence = vec![0_f32; config.fmt.sample_rate * 5];
    audio_samples.append(&mut silence);
    let detected_wakewords = audio_samples
        .chunks_exact(rustpotter.get_samples_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_samples(audio_buffer.into()))
        .map(|detection| {
            print_detection(&detection);
            detection
        })
        .collect::<Vec<_>>();
    detected_wakewords
}

fn run_detection_simulation(
    config: RustpotterConfig,
    model_path: &str,
) -> Vec<rustpotter::RustpotterDetection> {
    run_detection_simulation_with_gains(config, model_path, 1.0, 1.0)
}
fn run_detection_simulation_with_gains(
    mut config: RustpotterConfig,
    model_path: &str,
    sample_1_gain: f32,
    sample_2_gain: f32,
) -> Vec<rustpotter::RustpotterDetection> {
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_rate = 16000;
    let bits_per_sample = 16;
    config.fmt.sample_rate = sample_rate;
    config.fmt.sample_format = SampleFormat::I16;
    config.fmt.channels = 1;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + model_path;
    rustpotter
        .add_wakeword_from_file("wakeword", &model_path)
        .unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let live_audio_simulation = get_audio_with_two_wakewords_with_gain(
        sample_rate,
        bits_per_sample,
        sample_1_path,
        sample_2_path,
        sample_1_gain,
        sample_2_gain,
    );
    let detected_wakewords = live_audio_simulation
        .chunks_exact(rustpotter.get_bytes_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_bytes(audio_buffer))
        .map(|detection| {
            print_detection(&detection);
            detection
        })
        .collect::<Vec<_>>();
    detected_wakewords
}

fn get_audio_with_two_wakewords_with_gain(
    sample_rate: usize,
    bits_per_sample: u16,
    sample_1_path: String,
    sample_2_path: String,
    sample_1_gain: f32,
    sample_2_gain: f32,
) -> Vec<u8> {
    let mut live_audio_simulation: Vec<u8> = Vec::new();
    live_audio_simulation.append(&mut generate_silence_buffer(
        sample_rate,
        bits_per_sample,
        5,
    ));
    live_audio_simulation.append(&mut read_wav_buffer(&sample_1_path, sample_1_gain));
    live_audio_simulation.append(&mut generate_silence_buffer(
        sample_rate,
        bits_per_sample,
        5,
    ));
    live_audio_simulation.append(&mut read_wav_buffer(&sample_2_path, sample_2_gain));
    live_audio_simulation.append(&mut generate_silence_buffer(
        sample_rate,
        bits_per_sample,
        5,
    ));
    live_audio_simulation
}

fn generate_silence_buffer(sample_rate: usize, bit_depth: u16, seconds: usize) -> Vec<u8> {
    vec![0_u8; sample_rate * (bit_depth / 8) as usize * seconds]
}
// this function assumes wav file has i16 samples
fn read_wav_buffer(path: &str, gain: f32) -> Vec<u8> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    // remove wav header
    buffer.drain(0..44);
    let buffer_with_gain = buffer
        .chunks_exact(2)
        .map(|bytes| {
            i16::to_le_bytes(
                (i16::from_le_bytes([bytes[0], bytes[1]]) as f32 * gain)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            )
        })
        .fold(Vec::new(), |mut acc, b| {
            acc.append(&mut b.to_vec());
            acc
        });
    buffer_with_gain
}

fn print_detection(detection: &rustpotter::RustpotterDetection) {
    println!("-----=====-----");
    println!("Detection Score: {}", detection.score);
    println!("Avg Score: {}", detection.avg_score);
    println!("Scores: {:?}", detection.scores);
    println!("Partial detections: {}", detection.counter);
    println!("_______________");
}
