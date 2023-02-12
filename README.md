# Rustpotter

## An open source wakeword spotter forged in rust.

<div align="center">
    <img src="./logo.png?raw=true" width="400px"</img> 
</div>

## Description

The target of this project is to detect specific keywords on a live audio stream.

You can see Rustpotter a composition of the following tools:

* A `wav audio encoder`: Used to encode the input bytes into samples. Re-encodes the input samples to the internal format (16000Hz - 16bit - mono) if needed.
* A `gain normalization filter`: Used to reduce the input loudness to a level similar to the template, only when it's over a x2 factor. (Can be disabled)
* A `bass-pass filter`: Used to attenuate frequencies outside the configured range. (Can be disabled)
* A `feature extractor`: used to generate some feature frames for each encoded input sample chuck.
* A `feature comparator`: used to calculate de similarity between two feature matrix (two vectors of feature frames).

As a summary, Rustpotter keeps a window of feature frames that grows until been equal in length to the largest feature frame vector in the available wakewords.
From that moment it start to compare this window on each window update against each the wakewords feature frame vectors, in order to find a successful detection (max comparison score is over the defined `threshold`).

A detection is considered a `partial detection` (not emitted) until n more frames are processed (two times the size of the detector window).
If in this time a detection with a higher score is found, it replaces the current partial detection and the frame countdown is reset.

Note that the `cpu usage` will increase depending on the number of available feature frame vectors in the wakewords (the number of wav files used to created the wakeword instance).
In order to reduce the cpu usage, Rustpotter generates a single feature frame vector by averaging all the others ones. This one will score less that the others but can be used to skip the further comparison against each of the other feature frame vectors. This improvement is available whenever the `avg_threshold` option has a value greater that zero.

Using the `struct Wakeword`, the features can be extracted from wav files, persisted to a model file, or loaded from a previous generated model.
The detector supports adding multiple wakeword instances.
Note that Rustpotter `can not work with raw wav files`, as it will try to read the file wav format from its header.

A successful rustpotter detection looks like this:
```rust
RustpotterDetection {
    /// Detected wakeword name.
    name: "hey home",
    /// Detection score against the averaged feature frames. (zero if disabled)
    avg_score: 0.21601, 
    /// Detection score. (max of scores)
    score: 0.6618781, 
    /// Detection scores against each template.
    scores: {
        "hey_home_g_5.wav": 0.63050425, 
        "hey_home_g_3.wav": 0.6301979, 
        "hey_home_g_4.wav": 0.61404395, 
        "hey_home_g_1.wav": 0.6618781, 
        "hey_home_g_2.wav": 0.62885964
    },
    /// Partial detections counter.
    counter: 40
}
```

## Web Demos

 This [spot demo](https://givimad.github.io/rustpotter-worklet-demo/) is available so you can quickly try out Rustpotter using a web browser.

 It includes some models generated using multiple voices from a text-to-speech service.
 You can also load your own ones.

 This [model generator demo](https://givimad.github.io/rustpotter-create-model-demo/) is available so you can quickly generate rustpotter models using your own voice.

Please note that `both run entirely on your browser, your voice is not sent anywhere`.

## Related projects

* [rustpotter-cli](https://github.com/GiviMAD/rustpotter-cli): Use rustpotter on the `shell`.
* [rustpotter-java](https://github.com/GiviMAD/rustpotter-java): Use rustpotter on `java`.
* [rustpotter-wasm](https://github.com/GiviMAD/rustpotter-wasm): Generator for javascript + wasm module.
* [rustpotter-web](https://www.npmjs.com/package/rustpotter-web): Npm package generated with rustpotter-wasm targeting `web`.
* [rustpotter-worklet](https://github.com/GiviMAD/rustpotter-worklet): Ready to use package for `web`, simplifies `using Rustpotter as a Web Audio API node processor` (runs rustpotter-web using an AudioWorklet/ScriptProcessor depending on availability).

## Versioning

Rustpotter versions prior to v2.0.0 are not recommended.

Since 1.0.0 it will stick to [semver](https://semver.org), and a model compatibly break will be  marked by a MAJOR version change, same will apply for related packages (cli, wasm-wrapper, java-wrapper...).

## Basic Usage

```rust
use rustpotter::{Rustpotter, RustpotterConfig, Wakeword};
// assuming the audio input format match the detector defaults
let mut detector_config = RustpotterConfig::default();
// Configure the detector options
...
// Init the detector
let mut detector = Rustpotter::new(&detector_config).unwrap();
// load a wakeword
detector.add_wakeword_from_file("./tests/resources/hey_home_g.rpw").unwrap();
// You need a buffer of size `detector.get_samples_per_frame()` when using samples.
// You need a buffer of size `detector.get_bytes_per_frame()` when using bytes.
let mut frame_buffer: Vec<i16> = vec![0; detector.get_samples_per_frame()];
// while true { Iterate forever
    // fill the buffer with the required samples/bytes
    ...
    let detection_option = detector.process_short_int_buffer(frame_buffer);
    if detection_option.is_some() {
        let detection = detection_option..unwrap();
        println!("{:?}", detection);
    }
// }
```

### References

This project started as a port of the project [node-personal-wakeword](https://github.com/mathquis/node-personal-wakeword) and uses the method described in this medium [article](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e), and many other references, everything is based on shared knowledge online.

### Motivation

The motivation behind this project is to learn a little about audio analysis and Rust.
I have no prior experience with audio processing so don't expect this to be a production grade tool.

I found information about this keyword spotting method, and seems like a nice fit for a Rust project.

Feel encourage to suggest any improvements or fixes.

