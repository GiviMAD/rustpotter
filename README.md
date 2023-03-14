# Rustpotter

## An open source wakeword spotter forged in rust.

<div align="center">
    <img src="./logo.png?raw=true" width="400px"</img> 
</div>

## Description

The aim of this project is to detect specific keywords in a live audio stream.
Rustpotter is composed of the following tools:

* A `wav encoder`: Used to support various input formats and re-encode the samples to the internal format (16000Hz - 32bit - float - mono) when required.
* A `gain-normalizer filter`: This filter is used to dynamically change the input loudness based on a reference level (can be disabled).
* A `bass-pass filter`: This filter is used to attenuate frequencies outside the configured range (can be disabled).
* A `feature extractor`: This tool generates three feature vectors from each input chunk. These features are the [Mel Frequency Cepstral Coeﬃcients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of the audio.
* A `feature comparator`: This tool calculates the similarity between two sets of feature vectors (two matrices of features) using a [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm.


## Overview

In summary, when you feed Rustpotter with a stream, it keeps a window of feature vectors (can be seen as a matrix of features) that grows until it is equal in length to the largest matrix in the available wakewords.

The input length requested by Rustpotter varies depending on the configured format but is constant and equivalent to 30ms of audio. Internally, it generates a vector of features that represents 10ms of audio. So three new vectors are added on each execution.

From the moment the window has the correct size, Rustpotter starts comparing this window (on each new vector of features added) against each of the wakewords' feature matrices (those that represent the audio files that were used to create the wakeword), in order to find a successful detection (unifies the score based on the configured score mode and checks if it is over the defined `threshold`).

A detection is considered a `partial detection` (not emitted) until n more frames are processed (half of the length of the feature window). If in this time a detection with a higher score is found, it replaces the current partial detection, and this countdown is reset.

### Averaged Threshold

Note that Rustpotter's CPU usage increases with the number of WAV files used to create the wakeword, as the score operation runs against each of them.

To reduce the CPU usage, Rustpotter generates a single matrix of features for each wakeword by averaging all the others.
This one will score less than the others but can be used to skip the further comparisons against each of the matrix of features in the wakeword.

You can set the `avg_threshold` config to zero to disable this.

### Audio Filters

As described before, Rustpotter includes two audio filter implementations: a `gain-normalizer filter` and a `bass-pass filter`.

These filters are disabled by default, and their main purpose is to improve the detector's performance in the presence of noise.

### Partial detections

To discard false detections, you can require a certain number of partial detections to occur.
This is configured through the `min_scores` config option.

### Score Mode

As explained, Rustpotter scores the live audio matrix of features against each matrix of features in the wakeword. Then it needs to unify these scores into a single one.

You can configure how this is done using the `score_mode` option. The following modes are available:

* Avg: Use the averaged value (mean).
* Max: Use the maximum value.
* Median: Use the median. Equivalent to P50.
* P25, P50, P75, P80, P90, P95: Use the indicated percentile value. Linear interpolation between the values is used on non-exact matches.

### Wakeword

Using the `struct Wakeword`, you can extract the features from WAV files, persist them to a model file, or load them from a previously generated model.

Note that this `does not work with raw WAV files`; it parses the file format from its header.

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

As such, this is not intended to be a production-grade tool.

## Contributing

Feel free to suggest or contribute any improvements that you have in mind, either to the code or the detection process.
If you need any assistance, please feel free to open an issue.

Best regards!

