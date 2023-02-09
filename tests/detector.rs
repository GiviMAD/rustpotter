use std::{fs::File, io::{BufReader, Read}};

use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};

#[test]
fn it_can_detect_wakewords() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut config = RustpotterConfig::default();
    let sample_rate = 16000;
    config.fmt.sample_rate = sample_rate;
    config.fmt.bits_per_sample = 16;
    config.fmt.channels = 1;
    config.fmt.sample_format = SampleFormat::Int;
    config.avg_threshold = 0.2;
    config.threshold = 0.5;
    let mut rustpotter = Rustpotter::new(config).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    rustpotter.add_wakeword_from_file(&model_path).unwrap();
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    let mut live_audio_simulation: Vec<u8> = Vec::new();
    live_audio_simulation.append(&mut generate_silence_buffer(sample_rate, 5));
    live_audio_simulation.append(&mut read_wav_buffer(&sample_1_path));
    live_audio_simulation.append(&mut generate_silence_buffer(sample_rate, 5));
    live_audio_simulation.append(&mut read_wav_buffer(&sample_2_path));
    live_audio_simulation.append(&mut generate_silence_buffer(sample_rate, 5));
    let detected_wakewords = live_audio_simulation
    .chunks_exact(rustpotter.get_bytes_per_frame())
    .filter_map(|audio_buffer| rustpotter.process_byte_buffer(audio_buffer.to_vec()))
    .collect::<Vec<_>>();
    assert_eq!(detected_wakewords.len(), 2);
    assert_eq!(detected_wakewords[0].score, 0.72954464);
    assert_eq!(detected_wakewords[1].score, 0.703174);
    
}

fn generate_silence_buffer(sample_rate: usize, seconds: usize)  -> Vec<u8> {
    vec![
        0_u8;
        sample_rate * seconds
    ]
}
fn read_wav_buffer(path: &str) -> Vec<u8> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    // remove wav header
    buffer.drain(0..44);
    buffer
}