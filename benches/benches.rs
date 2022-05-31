#[macro_use]
extern crate bencher;

use std::{
    fs::File,
    io::{BufReader, Read},
};

use bencher::Bencher;

fn spot_wakewords(bench: &mut Bencher) {
    let mut detector = rustpotter::WakewordDetectorBuilder::new()
        .set_sample_rate(16000)
        .build();
    let dir = env!("CARGO_MANIFEST_DIR");
    let sample_1_path = dir.to_owned() + "/tests/resources/oye_casa_g_1.wav";
    let sample_2_path = dir.to_owned() + "/tests/resources/oye_casa_g_2.wav";
    detector.add_wakeword(
        "hey home",
        true,
        None,
        None,
        vec![sample_1_path.clone(), sample_2_path.clone()],
    );
    let mut audio_recreation: Vec<u8> = Vec::new();

    audio_recreation.append(&mut vec![0_u8; detector.get_bytes_per_frame() * 2]);
    let mut sample_1_bytes = read_buffer(File::open(sample_1_path).unwrap());
    audio_recreation.append(&mut sample_1_bytes);
    audio_recreation.append(&mut vec![0_u8; detector.get_bytes_per_frame() * 2]);
    let mut sample_2_bytes = read_buffer(File::open(sample_2_path).unwrap());
    audio_recreation.append(&mut sample_2_bytes);
    audio_recreation.append(&mut vec![0_u8; detector.get_bytes_per_frame() * 2]);
    bench.iter(|| {
        let _detections = audio_recreation
            .chunks_exact(detector.get_bytes_per_frame())
            .for_each(|audio_buffer| {
                detector.process_buffer(audio_buffer);
            });
    });
}
fn read_buffer(f: File) -> Vec<u8> {
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    buffer
}

benchmark_group!(benches, spot_wakewords);
benchmark_main!(benches);
