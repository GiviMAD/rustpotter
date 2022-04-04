# Rustpotter

## A personal keywords spotter written in Rust

<div align="center">
    <img src="./logo.png?raw=true" width="400px"</img> 
</div>

## Description

This project allows detect concrete words on and audio stream, to do so it generates a set of features from some word audio samples to later compare them with the features generated from a live audio stream, to calculate the probability of a match.

The features can be loaded from a previous generated model file or extracted from the samples before start the live streaming.

## Some examples:

### Create keyword model
```rust
let mut detector_builder = detector::FeatureDetectorBuilder::new();
    let mut word_detector = detector_builder.build();
    let name = String::from("hey home");
    let path = String::from("./hey_home.rpw");
    word_detector.add_keyword(
        name.clone(),
        false,
        true,
        None,
        vec!["./<audio sample path>.wav", "./<audio sample path>2.wav", ...],
    );
    match word_detector.create_wakeword_model(name.clone(), path) {
        Ok(_) => {
            println!("{} created!", name);
        }
        Err(message) => {
           panic!(message);
        }
    };
```


### Spot keyword:
```rust
    let mut detector_builder = detector::FeatureDetectorBuilder::new();
    detector_builder.set_threshold(0.4);
    let mut word_detector = detector_builder.build();
    let result = word_detector.add_keyword_from_model(command.model_path, command.average_templates, true, None);
    if result.is_err() {
        panic!("Unable to load keyword model");
    }
    while true {
        let mut frame_buffer = vec![0; word_detector.get_samples_per_frame()];
        let pcm_signed_buffer: Vec<i16> = ...;
         let detections = word_detector.process_pcm_signed(frame_buffer);
        for detection in detections {
            println!("Detected '{}' with score {}!", detection.wakeword, detection.score)
        }
    }

```

### References

This project is mostly a port of the project [node-personal-wakeword](https://github.com/mathquis/node-personal-wakeword) with some utils ported from [Gist](https://github.com/adamstark/Gist) so credit about the implementation is for those projects. Also to this medium [article](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e) about wake word detection 

### Motivation

The motivation behind this project is to learn about audio analysis and Rust, also to have access to an open source personal wakeword spotter to use in other home projects. Feel free to propose or PR any improvements or fixes.

