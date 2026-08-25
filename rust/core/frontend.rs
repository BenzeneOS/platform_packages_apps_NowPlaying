//! Fingerprint frontend matched to Google's production arithmetic.
//!
//! Production PCM16 is resampled to 11,025 Hz before analysis. Each two-second analysis window is
//! split into 42 overlapping 1,024-sample frames with a periodic
//! Hamming window. The FFT emits 513 magnitudes, then DC is discarded to leave bins 1 through
//! 512 for the embedder.
//!
//! The resampler uses the recovered high quality filter with separate float32 left and right
//! accumulators. Changing the filter or combining those accumulations changes the production
//! tensor.
//!
//! The window shape and `google_fft` operation order are load-bearing. Together they match all
//! 21,504 of Google's model inputs bit for bit. A conventional float64 FFT still left 14,647
//! values different, and a symmetric Hamming window also silently changes recognition results.

use crate::google_fft::GoogleFft;
use crate::resample::downsample_by_two;
use crate::{Error, Result};

/// Input PCM rate expected by [`Frontend::process_stream`], in hertz.
pub const SAMPLE_RATE: usize = 22_050;
/// Samples in one downsampled analysis window, covering two seconds of source PCM.
pub const OUTER_WINDOW: usize = 22_050;
/// Samples between downsampled analysis windows, advancing one source second.
pub const OUTER_STEP: usize = 11_025;
/// Samples transformed in each spectrogram frame.
pub const FFT_SIZE: usize = 1_024;
/// Samples between adjacent spectrogram frames.
pub const FFT_HOP: usize = 512;
/// Complete FFT frames produced from one analysis window.
pub const FRAME_COUNT: usize = 42;
/// Magnitude bins retained after dropping DC.
pub const BIN_COUNT: usize = 512;
/// Float count expected by the NNFP embedder for one analysis window.
pub const MODEL_INPUT_SIZE: usize = FRAME_COUNT * BIN_COUNT;

const PI: f64 = std::f64::consts::PI;

/// Intermediate buffers from one fingerprint frontend pass.
///
/// The expanded buffers make parity failures observable at the PCM, window, FFT, and magnitude
/// boundaries. Production recognition consumes only [`Self::model_input`].
#[derive(Debug, Clone)]
pub struct FrontendOutput {
    /// The 22,050 normalized float samples analyzed by this pass.
    pub pcm: Vec<f32>,
    /// Frame-major periodic-Hamming output with `42 * 1_024` values.
    pub windowed_frames: Vec<f32>,
    /// Frame-major real FFT output with `42 * 1_024` values.
    pub fft_real: Vec<f32>,
    /// Frame-major imaginary FFT output with `42 * 1_024` values.
    pub fft_imag: Vec<f32>,
    /// Frame-major magnitudes for all 513 real FFT bins, including DC.
    pub magnitudes: Vec<f32>,
    /// Frame-major bins 1 through 512 in the `42 * 512` layout required by the embedder.
    pub model_input: Vec<f32>,
}

/// Stateful workspace for producing the embedder's exact spectrogram input.
///
/// Reusing a value avoids allocating FFT scratch buffers for each frame. The public streaming
/// path also performs the recovered fixed half-rate resampling before framing.
#[derive(Debug, Clone)]
pub struct Frontend {
    window: Vec<f32>,
    fft: GoogleFft,
    input: Vec<f32>,
    real: Vec<f32>,
    imag: Vec<f32>,
    fft_work: Vec<f32>,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend {
    /// Allocates reusable buffers and the periodic 1,024-point Hamming window.
    pub fn new() -> Self {
        let window = (0..FFT_SIZE)
            .map(|index| (0.54 - 0.46 * (2.0 * PI * index as f64 / FFT_SIZE as f64).cos()) as f32)
            .collect();
        Self {
            window,
            fft: GoogleFft::new(),
            input: vec![0.0; FFT_SIZE],
            real: vec![0.0; FFT_SIZE],
            imag: vec![0.0; FFT_SIZE],
            fft_work: vec![0.0; FFT_SIZE],
        }
    }

    /// Analyzes exactly one PCM16 window without applying the production downsampler.
    ///
    /// This low-level parity entry point converts [`OUTER_WINDOW`] values to `[-1, 1)` and
    /// analyzes them directly. Use [`Self::process_stream`] for 22,050 Hz production audio.
    pub fn process(&mut self, samples: &[i16]) -> Result<FrontendOutput> {
        if samples.len() != OUTER_WINDOW {
            return Err(Error::InvalidInput(format!(
                "frontend needs {OUTER_WINDOW} PCM16 samples and received {}",
                samples.len()
            )));
        }

        let pcm = samples
            .iter()
            .map(|&sample| sample as f32 * 2.0f32.powi(-15))
            .collect::<Vec<_>>();
        self.process_f32(&pcm)
    }

    pub(crate) fn process_f32(&mut self, pcm: &[f32]) -> Result<FrontendOutput> {
        if pcm.len() != OUTER_WINDOW {
            return Err(Error::InvalidInput(format!(
                "frontend needs {OUTER_WINDOW} float samples and received {}",
                pcm.len()
            )));
        }
        let mut windowed_frames = Vec::with_capacity(FRAME_COUNT * FFT_SIZE);
        let mut fft_real = Vec::with_capacity(FRAME_COUNT * FFT_SIZE);
        let mut fft_imag = Vec::with_capacity(FRAME_COUNT * FFT_SIZE);
        let mut magnitudes = Vec::with_capacity(FRAME_COUNT * (BIN_COUNT + 1));
        let mut model_input = Vec::with_capacity(MODEL_INPUT_SIZE);

        for frame in 0..FRAME_COUNT {
            let start = frame * FFT_HOP;
            for index in 0..FFT_SIZE {
                self.input[index] = pcm[start + index] * self.window[index];
            }
            windowed_frames.extend_from_slice(&self.input);
            self.real.fill(0.0);
            self.imag.fill(0.0);
            self.fft.process(
                &self.input,
                &mut self.real,
                &mut self.imag,
                &mut self.fft_work,
            );
            fft_real.extend_from_slice(&self.real);
            fft_imag.extend_from_slice(&self.imag);

            magnitudes.push(self.real[0].abs());
            for bin in 1..BIN_COUNT {
                let magnitude =
                    (self.real[bin] * self.real[bin] + self.imag[bin] * self.imag[bin]).sqrt();
                magnitudes.push(magnitude);
                model_input.push(magnitude);
            }
            let nyquist = self.real[BIN_COUNT].abs();
            magnitudes.push(nyquist);
            model_input.push(nyquist);
        }

        Ok(FrontendOutput {
            pcm: pcm.to_vec(),
            windowed_frames,
            fft_real,
            fft_imag,
            magnitudes,
            model_input,
        })
    }

    /// Downsamples 22,050 Hz PCM16 and returns every complete overlapping frontend window.
    ///
    /// Windows cover two source seconds and advance one second. Inputs shorter than two seconds
    /// produce an empty vector rather than a partial embedding.
    pub fn process_stream(&mut self, samples: &[i16]) -> Result<Vec<FrontendOutput>> {
        let resampled = downsample_by_two(samples);
        self.process_resampled_stream(&resampled)
    }

    pub(crate) fn process_resampled_stream(
        &mut self,
        resampled: &[f32],
    ) -> Result<Vec<FrontendOutput>> {
        if resampled.len() < OUTER_WINDOW {
            return Ok(Vec::new());
        }
        let count = (resampled.len() - OUTER_WINDOW) / OUTER_STEP + 1;
        let mut outputs = Vec::with_capacity(count);
        for start in (0..=resampled.len() - OUTER_WINDOW).step_by(OUTER_STEP) {
            outputs.push(self.process_f32(&resampled[start..start + OUTER_WINDOW])?);
        }
        Ok(outputs)
    }
}

/// Generates the fixed PCM16 sequence used by the bit-parity frontend fixture.
pub fn deterministic_pcm() -> Vec<i16> {
    let mut state = 0x1234_5678u32;
    (0..OUTER_WINDOW)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_input_has_the_recovered_shape() {
        let output = Frontend::new().process(&deterministic_pcm()).unwrap();
        assert_eq!(output.pcm.len(), OUTER_WINDOW);
        assert_eq!(output.windowed_frames.len(), FRAME_COUNT * FFT_SIZE);
        assert_eq!(output.magnitudes.len(), FRAME_COUNT * (BIN_COUNT + 1));
        assert_eq!(output.model_input.len(), MODEL_INPUT_SIZE);
        assert!(output.model_input.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn matches_probe_frontend_fixture() {
        let expected = include_bytes!("../tests/fixtures/google-frontend.f32le")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        let output = Frontend::new().process(&deterministic_pcm()).unwrap();
        assert_eq!(output.model_input, expected);
    }

    #[test]
    fn stream_uses_the_recovered_outer_step() {
        let samples = vec![0; SAMPLE_RATE * 3];
        assert_eq!(Frontend::new().process_stream(&samples).unwrap().len(), 2);
    }

    #[test]
    fn production_call_emits_seven_inputs_from_eight_seconds() {
        let samples = vec![0; SAMPLE_RATE * 8];
        assert_eq!(Frontend::new().process_stream(&samples).unwrap().len(), 7);
    }
}
