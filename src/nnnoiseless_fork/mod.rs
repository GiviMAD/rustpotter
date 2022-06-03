//! NOTE: This file was forked temporally from the nnnoiseless project
//! Structures for computing audio features.
//!
//! This module contains utilities for computing features of an audio signal. These features are
//! used in two ways: they can be fed into a trained neural net for noise removal and speech
//! detection, or when the `train` feature is enabled then can be collected and used to train new
//! neural nets.

/// copied from nnnoiseless
use once_cell::sync::OnceCell;
mod fft;
mod pitch;
pub const FRAME_SIZE: usize = 120 << FRAME_SIZE_SHIFT;
const FREQ_SIZE: usize = FRAME_SIZE + 1;
const NB_BANDS: usize = 22;
const NB_FEATURES: usize = NB_BANDS + 3 * NB_DELTA_CEPS + 2;
const WINDOW_SIZE: usize = 2 * FRAME_SIZE;
const CEPS_MEM: usize = 8;
const NB_DELTA_CEPS: usize = 6;
const PITCH_MAX_PERIOD: usize = 768;
const PITCH_FRAME_SIZE: usize = 960;
const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;
pub const FRAME_SIZE_SHIFT: usize = 2;
type Complex = rustfft::num_complex::Complex32;
struct CommonState {
    window: [f32; WINDOW_SIZE],
    dct_table: [f32; NB_BANDS * NB_BANDS],
    sin_cos_table: [(f32, f32); WINDOW_SIZE / 2],
    wnorm: f32,
}

static COMMON: OnceCell<CommonState> = OnceCell::new();

fn common() -> &'static CommonState {
    if COMMON.get().is_none() {
        let pi = std::f64::consts::PI;
        let mut window = [0.0; WINDOW_SIZE];
        for i in 0..FRAME_SIZE {
            let sin = (0.5 * pi * (i as f64 + 0.5) / FRAME_SIZE as f64).sin();
            window[i] = (0.5 * pi * sin * sin).sin() as f32;
            window[WINDOW_SIZE - i - 1] = (0.5 * pi * sin * sin).sin() as f32;
        }
        let wnorm = 1_f32 / window.iter().map(|x| x * x).sum::<f32>();

        let mut dct_table = [0.0; NB_BANDS * NB_BANDS];
        for i in 0..NB_BANDS {
            for j in 0..NB_BANDS {
                dct_table[i * NB_BANDS + j] =
                    ((i as f64 + 0.5) * j as f64 * pi / NB_BANDS as f64).cos() as f32;
                if j == 0 {
                    dct_table[i * NB_BANDS + j] *= 0.5f32.sqrt();
                }
            }
        }

        let mut sin_cos_table = [(0.0, 0.0); WINDOW_SIZE / 2];
        fft::precompute_sin_cos_table(&mut sin_cos_table[..]);
        let _ = COMMON.set(CommonState {
            window,
            dct_table,
            sin_cos_table,
            wnorm,
        });
    }
    COMMON.get().unwrap()
}
fn sin_cos_table() -> &'static [(f32, f32)] {
    &common().sin_cos_table[..]
}
fn dct(out: &mut [f32], x: &[f32]) {
    let c = common();
    for (i, out_item) in out.iter_mut().enumerate().take(NB_BANDS) {
        let mut sum = 0.0;
        for (j, x_item) in x.iter().enumerate().take(NB_BANDS) {
            sum += x_item * c.dct_table[j * NB_BANDS + i];
        }
        *out_item = (sum as f64 * (2.0 / NB_BANDS as f64).sqrt()) as f32;
    }
}
pub const EBAND_5MS: [usize; 22] = [
    // 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// A basic high-pass filter.
pub const BIQUAD_HP: Biquad = Biquad {
    a: [-1.99599, 0.99600],
    b: [-2.0, 1.0],
};

/// A biquad filter.
///
/// Our convention here is that both sets of coefficients come with an implicit leading "1". To be
/// precise, if `x` is the input then this filter outputs `y` defined by
/// ```text
/// y[n] = x[n] + b[0] * x[n-1] + b[1] * x[n-2] - a[0] * y[n-1] - a[1] * y[n-2].
/// ```
#[derive(Default)]
pub struct Biquad {
    /// The auto-regressive coefficients.
    pub a: [f32; 2],
    /// The moving-average coefficients.
    pub b: [f32; 2],
}

impl Biquad {
    /// Apply this biquad filter to `input`, putting the result in `output`.
    ///
    /// `mem` is a scratch buffer allowing you filter a long signal one buffer at a time. If you
    /// call this function multiple times with the same `mem` buffer, the output will be as though
    /// you had called it once with a longer `input`. The first time you call `filter` on a given
    /// signal, `mem` should be zero.
    pub fn filter(&self, output: &mut [f32], mem: &mut [f32; 2], input: &[f32]) {
        let a0 = self.a[0] as f64;
        let a1 = self.a[1] as f64;
        let b0 = self.b[0] as f64;
        let b1 = self.b[1] as f64;
        for (&x, y) in input.iter().zip(output) {
            let x64 = x as f64;
            let y64 = x64 + mem[0] as f64;
            mem[0] = (mem[1] as f64 + (b0 * x64 - a0 * y64)) as f32;
            mem[1] = (b1 * x64 - a1 * y64) as f32;
            *y = y64 as f32;
        }
    }

    /// Apply this biquad filter to `data`, modifying it in place.
    ///
    /// See [`Biquad::filter`] for more details.
    // This is only used when the "train" feature is active.
    #[cfg_attr(not(feature = "train"), allow(dead_code))]
    pub fn filter_in_place(&self, data: &mut [f32], mem: &mut [f32; 2]) {
        let a0 = self.a[0] as f64;
        let a1 = self.a[1] as f64;
        let b0 = self.b[0] as f64;
        let b1 = self.b[1] as f64;
        for x in data {
            let x64 = *x as f64;
            let y64 = x64 + mem[0] as f64;
            mem[0] = (mem[1] as f64 + (b0 * x64 - a0 * y64)) as f32;
            mem[1] = (b1 * x64 - a1 * y64) as f32;
            *x = y64 as f32;
        }
    }
}
fn compute_band_corr(out: &mut [f32], x: &[Complex], p: &[Complex]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }

    for i in 0..(NB_BANDS - 1) {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) << FRAME_SIZE_SHIFT;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = (EBAND_5MS[i] << FRAME_SIZE_SHIFT) + j;
            let corr = x[idx].re * p[idx].re + x[idx].im * p[idx].im;
            out[i] += (1.0 - frac) * corr;
            out[i + 1] += frac * corr;
        }
    }
    out[0] *= 2.0;
    out[NB_BANDS - 1] *= 2.0;
}

fn zip3<I, J, K>(i: I, j: J, k: K) -> impl Iterator<Item = (I::Item, J::Item, K::Item)>
where
    I: IntoIterator,
    J: IntoIterator,
    K: IntoIterator,
{
    i.into_iter()
        .zip(j.into_iter().zip(k))
        .map(|(x, (y, z))| (x, y, z))
}
fn apply_window(output: &mut [f32], input: &[f32]) {
    let c = common();
    for (x, &y, &w) in zip3(output, input, &c.window[..]) {
        *x = y * w;
    }
}

/// Contains the necessary state to compute the features of audio input, and synthesize the output.
#[derive(Clone)]
pub struct DenoiseFeatures {
    /// This stores some of the previous input. Currently, whenever we get new input we shift this
    /// backwards and copy the new input at the end. It might be worth investigating a ring buffer.
    input_mem: [f32; max(FRAME_SIZE, PITCH_BUF_SIZE)],
    /// This is some sort of ring buffer, storing the last bunch of cepstra.
    cepstral_mem: [[f32; NB_BANDS]; CEPS_MEM],
    /// The index pointing to the most recent cepstrum in `cepstral_mem`. The previous cepstra are
    /// at indices mem_id - 1, mem_id - 2, etc (wrapped appropriately).
    mem_id: usize,
    mem_hp_x: [f32; 2],
    fft: fft::RealFft,
    window_buf: [f32; WINDOW_SIZE],

    // What follows are various buffers. The names are cryptic, but they follow a pattern.
    /// The Fourier transform of the most recent frame of input.
    pub x: [Complex; FREQ_SIZE],
    /// The Fourier transform of a pitch-period-shifted window of input.
    pub p: [Complex; FREQ_SIZE],
    /// The band energies of `x`.
    pub ex: [f32; NB_BANDS],
    /// The band energies of `p`.
    pub ep: [f32; NB_BANDS],
    /// The band correlations between `x` and `p`.
    pub exp: [f32; NB_BANDS],
    /// The computed features.
    features: [f32; NB_FEATURES],

    pitch_finder: pitch::PitchFinder,
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

impl DenoiseFeatures {
    /// Creates a new, empty, `DenoiseFeatures`.
    pub fn new() -> DenoiseFeatures {
        DenoiseFeatures {
            input_mem: [0.0; max(FRAME_SIZE, PITCH_BUF_SIZE)],
            cepstral_mem: [[0.0; NB_BANDS]; CEPS_MEM],
            mem_id: 0,
            mem_hp_x: [0.0; 2],
            fft: fft::RealFft::new(sin_cos_table()),
            window_buf: [0.0; WINDOW_SIZE],
            x: [Complex::from(0.0); FREQ_SIZE],
            p: [Complex::from(0.0); FREQ_SIZE],
            ex: [0.0; NB_BANDS],
            ep: [0.0; NB_BANDS],
            exp: [0.0; NB_BANDS],
            features: [0.0; NB_FEATURES],
            pitch_finder: pitch::PitchFinder::new(),
        }
    }

    /// Returns the computed features.
    pub fn features(&self) -> &[f32] {
        &self.features[..]
    }

    /// Shifts our input buffer and adds the new input to it. This is only used for generating
    /// training data, because in normal use we apply a biquad filter while adding the new input.
    pub fn shift_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let new_idx = self.input_mem.len() - FRAME_SIZE;
        for i in 0..new_idx {
            self.input_mem[i] = self.input_mem[i + FRAME_SIZE];
        }
        for (x, y) in self.input_mem[new_idx..].iter_mut().zip(input) {
            *x = *y;
        }
    }

    /// Shifts our input buffer and adds the new input to it, while running the input through a
    /// high-pass filter.
    pub fn shift_and_filter_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let new_idx = self.input_mem.len() - FRAME_SIZE;
        for i in 0..new_idx {
            self.input_mem[i] = self.input_mem[i + FRAME_SIZE];
        }
        BIQUAD_HP.filter(&mut self.input_mem[new_idx..], &mut self.mem_hp_x, input);
    }

    fn find_pitch(&mut self) -> usize {
        let input = &self.input_mem[self.input_mem.len().checked_sub(PITCH_BUF_SIZE).unwrap()..];
        let (pitch, _gain) = self.pitch_finder.process(input);
        pitch
    }

    /// Computes the features of the current frame.
    ///
    /// - `x` is the Fourier transform of the input, and `ex` are its band energies
    /// - `p` is the Fourier transform of older input, with a lag of the pitch period; `ep` are its band
    ///     energies
    /// - `exp` is the band correlation between `x` and `p`
    /// - `features` are all the features of that get input to the neural network.
    ///
    /// The returns the noise level.
    pub fn compute_frame_features(&mut self) -> f32 {
        let mut ly = [0.0; NB_BANDS];
        let mut tmp = [0.0; NB_BANDS];

        transform_input(
            &mut self.fft,
            &self.input_mem,
            0,
            &mut self.window_buf,
            &mut self.x,
            &mut self.ex,
        );
        let pitch_idx = self.find_pitch();

        transform_input(
            &mut self.fft,
            &self.input_mem,
            pitch_idx,
            &mut self.window_buf,
            &mut self.p,
            &mut self.ep,
        );
        compute_band_corr(&mut self.exp[..], &self.x[..], &self.p[..]);
        for i in 0..NB_BANDS {
            self.exp[i] /= (0.001 + self.ex[i] * self.ep[i]).sqrt();
        }
        dct(&mut tmp[..], &self.exp[..]);
        for (i, tmp_item) in tmp.iter().enumerate().take(NB_DELTA_CEPS) {
            self.features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = *tmp_item;
        }

        self.features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
        self.features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
        self.features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);
        let mut log_max = -2.0;
        let mut follow = -2.0;
        let mut e = 0.0;
        for (i, ly_item) in ly.iter_mut().enumerate().take(NB_BANDS) {
            *ly_item = (1e-2 + self.ex[i])
                .log10()
                .max(log_max - 7.0)
                .max(follow - 1.5);
            log_max = log_max.max(*ly_item);
            follow = (follow - 1.5).max(*ly_item);
            e += self.ex[i];
        }

        if e < 0.04 {
            /* If there's no audio, avoid messing up the state. */
            for i in 0..NB_FEATURES {
                self.features[i] = 0.0;
            }
            return e;
        }
        dct(&mut self.features, &ly[..]);
        self.features[0] -= 12.0;
        self.features[1] -= 4.0;
        let ceps_0_idx = self.mem_id;
        let ceps_1_idx = if self.mem_id < 1 {
            CEPS_MEM + self.mem_id - 1
        } else {
            self.mem_id - 1
        };
        let ceps_2_idx = if self.mem_id < 2 {
            CEPS_MEM + self.mem_id - 2
        } else {
            self.mem_id - 2
        };

        for i in 0..NB_BANDS {
            self.cepstral_mem[ceps_0_idx][i] = self.features[i];
        }
        self.mem_id += 1;

        let ceps_0 = &self.cepstral_mem[ceps_0_idx];
        let ceps_1 = &self.cepstral_mem[ceps_1_idx];
        let ceps_2 = &self.cepstral_mem[ceps_2_idx];
        for i in 0..NB_DELTA_CEPS {
            self.features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
            self.features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
            self.features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0 * ceps_1[i] + ceps_2[i];
        }

        /* Spectral variability features. */
        let mut spec_variability = 0.0;
        if self.mem_id == CEPS_MEM {
            self.mem_id = 0;
        }
        for i in 0..CEPS_MEM {
            let mut min_dist = 1e15f32;
            for j in 0..CEPS_MEM {
                let mut dist = 0.0;
                for k in 0..NB_BANDS {
                    let tmp = self.cepstral_mem[i][k] - self.cepstral_mem[j][k];
                    dist += tmp * tmp;
                }
                if j != i {
                    min_dist = min_dist.min(dist);
                }
            }
            spec_variability += min_dist;
        }

        self.features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;

        e
    }    
}

/// Fourier transforms the input.
///
/// The Fourier transform goes in `x` and the band energies go in `ex`.
fn transform_input(
    fft: &mut fft::RealFft,
    input: &[f32],
    lag: usize,
    window_buf: &mut [f32; WINDOW_SIZE],
    x: &mut [Complex],
    ex: &mut [f32],
) {
    let input = &input[input.len().checked_sub(WINDOW_SIZE + lag).unwrap()..];
    apply_window(&mut window_buf[..], input);
    fft.forward(window_buf, x);

    // In the original RNNoise code, the forward transform is normalized and the inverse
    // tranform isn't. `rustfft` doesn't normalize either one, so we do it ourselves.
    let norm = common().wnorm;
    for x in &mut x[..] {
        *x *= norm;
    }

    compute_band_corr(ex, x, x);
}
