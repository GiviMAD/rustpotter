use std::{
    fs::File,
    io::{BufReader, Read},
};

use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};

#[test]
fn it_can_detect_wakewords() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut config = RustpotterConfig::default();
    let sample_rate = 16000;
    let bits_per_sample = 16;
    config.fmt.sample_rate = sample_rate;
    config.fmt.bits_per_sample = bits_per_sample;
    config.fmt.channels = 1;
    config.fmt.sample_format = SampleFormat::Int;
    config.detector.avg_threshold = 0.2;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer = false;
    config.filters.band_pass = false;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    rustpotter.add_wakeword_from_file(&model_path).unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let live_audio_simulation =
        get_audio_with_two_wakewords(sample_rate, bits_per_sample, sample_1_path, sample_2_path);
    let detected_wakewords = live_audio_simulation
        .chunks_exact(rustpotter.get_bytes_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_byte_buffer(audio_buffer))
        .collect::<Vec<_>>();
    assert_eq!(detected_wakewords.len(), 2);
    print_detection(&detected_wakewords[0]);
    assert_eq!(detected_wakewords[0].avg_score, 0.30797324);
    assert_eq!(detected_wakewords[0].score, 0.72954464);
    print_detection(&detected_wakewords[1]);
    assert_eq!(detected_wakewords[1].avg_score, 0.41669166);
    assert_eq!(detected_wakewords[1].score, 0.703174);
}

#[test]
fn it_can_detect_wakewords_while_applying_band_pass_audio_filter() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut config = RustpotterConfig::default();
    let sample_rate = 16000;
    let bits_per_sample = 16;
    config.fmt.sample_rate = sample_rate;
    config.fmt.bits_per_sample = bits_per_sample;
    config.fmt.channels = 1;
    config.fmt.sample_format = SampleFormat::Int;
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer = false;
    config.filters.band_pass = true;
    config.filters.low_cutoff = 80.0;
    config.filters.high_cutoff = 400.0;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    rustpotter.add_wakeword_from_file(&model_path).unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let live_audio_simulation =
        get_audio_with_two_wakewords(sample_rate, bits_per_sample, sample_1_path, sample_2_path);
    let detected_wakewords = live_audio_simulation
        .chunks_exact(rustpotter.get_bytes_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_byte_buffer(audio_buffer))
        .collect::<Vec<_>>();
    assert_eq!(detected_wakewords.len(), 2);
    print_detection(&detected_wakewords[0]);
    assert_eq!(detected_wakewords[0].score, 0.6618781);
    print_detection(&detected_wakewords[1]);
    assert_eq!(detected_wakewords[1].score, 0.6807945);
}

#[test]
fn it_can_detect_wakewords_while_applying_gain_normalizer_audio_filter() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut config = RustpotterConfig::default();
    let sample_rate = 16000;
    let bits_per_sample = 16;
    config.fmt.sample_rate = sample_rate;
    config.fmt.bits_per_sample = bits_per_sample;
    config.fmt.channels = 1;
    config.fmt.sample_format = SampleFormat::Int;
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer = true;
    config.filters.band_pass = false;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    rustpotter.add_wakeword_from_file(&model_path).unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let live_audio_simulation = get_audio_with_two_wakewords_with_gain(
        sample_rate,
        bits_per_sample,
        sample_1_path,
        sample_2_path,
        3.5,
        30.3,
    );
    let detected_wakewords = live_audio_simulation
        .chunks_exact(rustpotter.get_bytes_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_byte_buffer(audio_buffer))
        .collect::<Vec<_>>();
    assert_eq!(detected_wakewords.len(), 2);
    print_detection(&detected_wakewords[0]);
    assert_eq!(detected_wakewords[0].score, 0.7247474);
    print_detection(&detected_wakewords[1]);
    assert_eq!(detected_wakewords[1].score, 0.61788327);
}

#[test]
fn it_can_detect_wakewords_while_applying_band_pass_and_gain_normalizer_audio_filters() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut config = RustpotterConfig::default();
    let sample_rate = 16000;
    let bits_per_sample = 16;
    config.fmt.sample_rate = sample_rate;
    config.fmt.bits_per_sample = bits_per_sample;
    config.fmt.channels = 1;
    config.fmt.sample_format = SampleFormat::Int;
    config.detector.avg_threshold = 0.;
    config.detector.threshold = 0.5;
    config.filters.gain_normalizer = true;
    config.filters.band_pass = true;
    config.filters.low_cutoff = 80.0;
    config.filters.high_cutoff = 400.0;
    let mut rustpotter = Rustpotter::new(&config).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    rustpotter.add_wakeword_from_file(&model_path).unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let live_audio_simulation = get_audio_with_two_wakewords_with_gain(
        sample_rate,
        bits_per_sample,
        sample_1_path,
        sample_2_path,
        3.5,
        30.3,
    );
    let detected_wakewords = live_audio_simulation
        .chunks_exact(rustpotter.get_bytes_per_frame())
        .filter_map(|audio_buffer| rustpotter.process_byte_buffer(audio_buffer))
        .collect::<Vec<_>>();
    assert_eq!(detected_wakewords.len(), 2);
    print_detection(&detected_wakewords[0]);
    assert_eq!(detected_wakewords[0].score, 0.6476441);
    print_detection(&detected_wakewords[1]);
    assert_eq!(detected_wakewords[1].score, 0.61747557);
}

fn get_audio_with_two_wakewords(
    sample_rate: usize,
    bits_per_sample: u16,
    sample_1_path: String,
    sample_2_path: String,
) -> Vec<u8> {
    get_audio_with_two_wakewords_with_gain(
        sample_rate,
        bits_per_sample,
        sample_1_path,
        sample_2_path,
        1.,
        1.,
    )
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
            i16::to_le_bytes((i16::from_le_bytes([bytes[0], bytes[1]]) as f32 * gain) as i16)
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
