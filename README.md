# Rustpotter

## A free and open source wake word spotter forged in rust

<div align="center">
    <img src="./logo.png?raw=true" width="400px"</img> 
</div>

## Warning!

This project is under active development and testing, api and model format are broken usually, consider rustpotter and all the related projects in beta until version 1.0.0 is published.

But don't hesitate to report any issues in the meantime.

## Description

This project allows to detect specific wake words on a live audio stream, to do so it generates a set of features from some audio samples to later compare them with the features generated from a live stream, to calculate the probability of a match.

The features can be loaded from a previous generated model file or extracted from the samples before start the live streaming.

## Related projects

* [rustpotter-cli](https://github.com/GiviMAD/rustpotter-cli): Use rustpotter on the command line (the only that exposes model generation at the moment).
* [rustpotter-java](https://github.com/GiviMAD/rustpotter-java): Use rustpotter on java.
* [rustpotter-wasm](https://github.com/GiviMAD/rustpotter-wasm): Generator for javascript + wasm module.
* [rustpotter-worklet](https://github.com/GiviMAD/rustpotter-worklet): Ready to use package for web (runs rustpotter-web in a worklet).
* [rustpotter-web](https://www.npmjs.com/package/rustpotter-web): Npm package generated with rustpotter-wasm targeting web.

## Versioning

All related packages use the version number of the rustpotter dependency (PATCH number could defer) so models generated with some version of the cli should work with other packages on the same version.

## Some examples:

### Create wakeword model:
```rust
let mut detector_builder = detector::FeatureDetectorBuilder::new();
    let mut word_detector = detector_builder.build();
    let name = String::from("hey home");
    let path = String::from("./hey_home.rpw");
    word_detector.add_wakeword(
        name.clone(),
        false,
        None,
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


### Spot wakeword:
```rust
    let mut detector_builder = detector::FeatureDetectorBuilder::new();
    detector_builder.set_threshold(0.4);
    detector_builder.set_sample_rate(16000);
    let mut word_detector = detector_builder.build();
    let result = word_detector.add_wakeword_from_model(command.model_path, command.average_templates, true, None);
    if result.is_err() {
        panic!("Unable to load wakeword model");
    }
    while true {
        let mut frame_buffer: Vec<i16> = vec![0; word_detector.get_samples_per_frame()];
        // fill the buffer
        ...
        let detection = word_detector.process_pcm_signed(frame_buffer);
        if detection.is_some() {
            println!("Detected '{}' with score {}!", detection.unwrap().wakeword, detection.unwrap().score)
        }
    }

```

### References

This project started as a port of the project [node-personal-wakeword](https://github.com/mathquis/node-personal-wakeword) and uses the method described in this medium [article](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e).

### Motivation

The motivation behind this project is to learn about audio analysis and Rust.
Also to have access to an open source personal wakeword spotter to use in other open projects.

Feel free to suggest any improvements or fixes.

