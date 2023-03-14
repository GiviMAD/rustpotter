# Rustpotter

## An open source wakeword spotter forged in rust.

<div align="center">
    <img src="./logo.png?raw=true" width="400px"</img> 
</div>

## Description

The target of this project is to detect specific keywords on a live audio stream.

You can see Rustpotter a composition of the following tools:

* A `wav encoder`: Used to support different the input formats, re-encodes the samples to the internal format (16000Hz - 32bit - float - mono) when needed.
* A `gain-normalizer filter`: Used to dynamically change the input loudness in base to a reference level. (Can be disabled)
* A `bass-pass filter`: Used to attenuate frequencies outside the configured range. (Can be disabled)
* A `feature extractor`: used to generate 3 vector of features from each input chuck. This called features are the [Mel Frequency Cepstral Coeﬃcients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of the audio.
* A `feature comparator`: used to calculate de similarity between two sets of feature vectors (two matrix of features) using a [DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm.

## Overview

As a summary, when you feed Rustpotter with a stream it keeps a window of feature vectors (can be seen as a matrix of features) that grows until been equal in length to the largest in the available wakewords.

The input length requested by Rustpotter varies depending on the configured format but it's constant and equivalent to 30ms of audio. Internally it generates a vector of features that represent 10ms of audio. So 3 new vectors are added on each execution.

From the moment the window has the correct size, Rustpotter start to compare this window (on each new vector of features added) against each of the wakewords feature matrixes (those are the ones that represent the audio files that you used to create the wakeword), in order to find a successful detection (unifies the score based on the configured score mode and check if it's over the defined `threshold`).

A detection is considered a `partial detection` (not emitted) until n more frames are processed (half of the length of the features window).
If in this time a detection with a higher score is found, it replaces the current partial detection and the this countdown is reset.

### Averaged Threshold

Note that Rustpotter `cpu usage` increases depending on the number of available feature frame vectors in the wakewords (the number of wav files used to create the wakeword), as for each new features Rustpotter has to run the score operation against each of them.

In order to reduce the cpu usage, for each wakeword, Rustpotter generates a `single matrix of features by averaging all` the others ones. This one will score less that the others but can be used to skip the further comparison against the other matrixes of features inside the wakeword. You can set the `avg_threshold` config to zero to disable this.

### Audio Filters

As described before rustpotter includes two audio filter implementation: a `gain-normalizer filter` and a `bass-pass filter`.

Those are disabled by default and their main purpose is to improve the detector performance on presence of noise.

### Partial detections

In order to discard false detections you can require a certain amount of partial detections to occur. This is configured through the `min_scores` config option.

### Score Mode

As explained rustpotter scores the live audio matrix of features against each of matrix of features in the wakeword. The it needs to unify these scores into a single one.

You configure how this is done using the `score_mode` option, the following modes are available:

* Avg: Use the averaged/mean value.
* Max: Use max value.
* Median: Use the median. Equivalent to P50.
* P25, P50, P75, P80, P90, P95: Use the indicated percentile value. Linear interpolation between the values is used on non exact matches.

### Wakeword

Using the `struct Wakeword`, the features can be extracted from wav files, persisted to a model file, or loaded from a previous generated model.

Note that this `doesn't work with raw wav files`, it parses the file format from its header.

The detector supports adding multiple wakewords. 

### Detection

A successful Rustpotter detection provides you with all the relevant information about the detection process so you know how to configure the detector to achieve a good configuration (minimize the number of misses/false detections).

It looks like this:

```rust
RustpotterDetection {
    /// Detected wakeword name.
    name: "hey home",
    /// Detection score against the averaged features matrix. (zero if disabled)
    avg_score: 0.41601, 
    /// Detection score. (calculated from the scores using the selected score mode).
    score: 0.6618781, 
    /// Detection score against each template.
    scores: {
        "hey_home_g_5.wav": 0.63050425, 
        "hey_home_g_3.wav": 0.6301979, 
        "hey_home_g_4.wav": 0.61404395, 
        "hey_home_g_1.wav": 0.6618781, 
        "hey_home_g_2.wav": 0.62885964
    },
    /// Number of partial detections.
    counter: 40,
    /// Gain applied by the gain-normalizer or 1.
    gain: 1.,
}
```

Rustpotter exposes a reference to the current partial detection that allows read access to it for debugging purposes.

## Web Demos

 The [spot demo](https://givimad.github.io/rustpotter-worklet-demo/) is available so you can quickly try out Rustpotter using a web browser.

 It includes some models generated using multiple voices from a text-to-speech service.
 You can also load your own ones.

 The [model generator demo](https://givimad.github.io/rustpotter-create-model-demo/) is available so you can quickly record samples and generate Rustpotter models using your own voice.

Please note that `both run entirely on your browser, your voice is not sent anywhere`, they are hosted using Github Pages.

## Related projects

* [rustpotter-cli](https://github.com/GiviMAD/rustpotter-cli): Use Rustpotter on the `shell`. (Window, macOs and Linux).
* [rustpotter-java](https://github.com/GiviMAD/rustpotter-java): Use Rustpotter on `java`. (Mvn package and generator)
* [rustpotter-wasm](https://github.com/GiviMAD/rustpotter-wasm): Generator for javascript + wasm module.
* [rustpotter-web](https://www.npmjs.com/package/rustpotter-web): Use Rustpotter on the `web`. (Npm package generated with rustpotter-wasm)
* [rustpotter-worklet](https://github.com/GiviMAD/rustpotter-worklet): Use Rustpotter as a `Web Audio API node processor`. (Runs rustpotter-web using an AudioWorklet/ScriptProcessor depending on availability)

## Versioning

Rustpotter versions prior to v2.0.0 are not recommended, this version was started from scratch reusing some code.

Since 1.0.0 it will stick to [semver](https://semver.org), and a model compatibly break will be  marked by a MAJOR version change, same will apply for related packages (cli, wasm-wrapper, java-wrapper...).

## Basic Usage

```rust
use rustpotter::{Rustpotter, RustpotterConfig, Wakeword};
// assuming the audio input format match the rustpotter defaults
let mut rustpotter_config = RustpotterConfig::default();
// Configure format/filters/detection options
...
// Instantiate rustpotter
let mut rustpotter = Rustpotter::new(&rustpotter_config).unwrap();
// load a wakeword
rustpotter.add_wakeword_from_file("./tests/resources/hey_home_g.rpw").unwrap();
// You need a buffer of size `rustpotter.get_samples_per_frame()` when using samples.
// You need a buffer of size `rustpotter.get_bytes_per_frame()` when using bytes.
let mut sample_buffer: Vec<i16> = vec![0; rustpotter.get_samples_per_frame()];
// while true { Iterate forever
    // fill the buffer with the required samples/bytes
    ...
    let detection = rustpotter.process_i16(sample_buffer);
    if let Some(detection) = detection {
        println!("{:?}", detection);
    }
// }
```

## References

This project started as a port of the project [node-personal-wakeword](https://github.com/mathquis/node-personal-wakeword) and uses the method described in this [medium article](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e).

## Motivation

The motivation behind this project is to learn about audio analysis and the Rust language/ecosystem.

I have no prior experience with any of those so don't expect this to be a production grade tool.

## Contributing

Feel encourage to suggest/contribute any improvements that you have in mind, about the code or the detection process.

Open an issue if you need any assistance about it.

Best regards!

