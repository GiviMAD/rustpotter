use std::{
    fs::File,
    io::{BufReader, Read},
};
static INIT_LOGGER: std::sync::Once = std::sync::Once::new();

use log::info;
use rustpotter::WakewordDetectorBuilder;

pub fn enable_rustpotter_log() {
    INIT_LOGGER.call_once(|| {
        simple_logger::SimpleLogger::new()
            .with_level(log::LevelFilter::Debug)
            .init()
            .unwrap()
    });
}
#[test]
fn it_returns_correct_samples_per_frame() {
    enable_rustpotter_log();
    let detector = WakewordDetectorBuilder::new().build();
    assert_eq!(480, detector.get_samples_per_frame());
}
#[test]
fn it_returns_correct_samples_per_frame_when_resampling() {
    let detector = WakewordDetectorBuilder::new()
        .set_sample_rate(16000)
        .build();
    assert_eq!(160, detector.get_samples_per_frame());
}
#[test]
fn it_returns_correct_frame_byte_length() {
    let detector = WakewordDetectorBuilder::new().build();
    assert_eq!(960, detector.get_bytes_per_frame());
}
#[test]
fn it_returns_correct_frame_byte_length_when_resampling() {
    let detector = WakewordDetectorBuilder::new()
        .set_sample_rate(16000)
        .build();
    assert_eq!(320, detector.get_bytes_per_frame());
}
#[test]
fn it_can_add_wakeword_from_samples() {
    enable_rustpotter_log();
    let dir = env!("CARGO_MANIFEST_DIR");
    let samples = vec![
        dir.to_owned() + "/tests/resources/oye_casa_g_1.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_2.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_3.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_4.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_5.wav",
    ];
    let mut detector = WakewordDetectorBuilder::new().build();
    detector.add_wakeword_with_wav_files("oye casa", true, None, None, samples).unwrap();
    detector
        .generate_wakeword_model_file(
            "oye casa".to_owned(),
            dir.to_owned() + "/tests/resources/oye_casa.rpw",
        )
        .unwrap();
}

#[test]
fn it_can_add_wakeword_from_model() {
    enable_rustpotter_log();
    let mut detector = WakewordDetectorBuilder::new().build();
    let dir = env!("CARGO_MANIFEST_DIR");
    let result = detector
        .add_wakeword_from_model_file(dir.to_owned() + "/tests/resources/oye_casa.rpw", true);
    assert!(result.is_ok());
}

#[test]
fn it_can_spot_wakewords() {
    enable_rustpotter_log();
    can_spot_wakewords_test_impl(&mut WakewordDetectorBuilder::new());
}
#[test]
fn it_can_spot_wakewords_in_eager_mode() {
    enable_rustpotter_log();
    can_spot_wakewords_test_impl(WakewordDetectorBuilder::new().set_eager_mode(true));
}
#[test]
fn it_can_spot_wakewords_while_detecting_noise() {
    enable_rustpotter_log();
    can_spot_wakewords_with_silence_frames_test_impl(
        WakewordDetectorBuilder::new().set_noise_mode(rustpotter::NoiseDetectionMode::Normal),
        1000,
    );
}

// utils
fn can_spot_wakewords_test_impl(builder: &mut WakewordDetectorBuilder) {
    can_spot_wakewords_with_silence_frames_test_impl(builder, 100);
}
fn can_spot_wakewords_with_silence_frames_test_impl(
    builder: &mut WakewordDetectorBuilder,
    silence_frames: usize,
) {
    enable_rustpotter_log();
    let mut detector = builder.set_sample_rate(16000).build();
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    detector.add_wakeword_with_wav_files(
        "hey home",
        true,
        Some(0.3),
        Some(0.67),
        vec![sample_1_path.clone(), sample_2_path.clone()],
    ).unwrap();
    let mut audio_recreation: Vec<u8> = Vec::new();
    audio_recreation.append(&mut vec![
        0_u8;
        detector.get_bytes_per_frame() * silence_frames
    ]);
    let mut sample_1_bytes = read_wav_buffer(File::open(sample_1_path).unwrap());
    audio_recreation.append(&mut sample_1_bytes);
    audio_recreation.append(&mut vec![
        0_u8;
        detector.get_bytes_per_frame() * silence_frames
    ]);
    let mut sample_2_bytes = read_wav_buffer(File::open(sample_2_path).unwrap());
    audio_recreation.append(&mut sample_2_bytes);
    audio_recreation.append(&mut vec![
        0_u8;
        detector.get_bytes_per_frame() * silence_frames
    ]);
    let detections = audio_recreation
        .chunks_exact(detector.get_bytes_per_frame())
        .filter_map(|audio_buffer| detector.process_buffer(audio_buffer))
        .collect::<Vec<_>>();
    assert_eq!(detections.len(), 2);
    for det in detections {
        info!("detection {}", det.score);
    }
}
fn read_wav_buffer(f: File) -> Vec<u8> {
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    // remove wav header
    buffer.drain(0..44);
    buffer
}
