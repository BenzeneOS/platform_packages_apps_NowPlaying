//! Frontend for the five-second music detector window.
//!
//! Production PCM16 is resampled to 11,025 Hz. Each five-second window advances by one second and
//! produces 214 frames using a periodic 512-point Hamming window with a 256-sample hop. All 257
//! real FFT magnitudes are retained, including DC and Nyquist.

use std::f64::consts::PI;

use crate::google_fft::GoogleFft;
use crate::music_detector::{MUSIC_INPUT_HEIGHT, MUSIC_INPUT_SIZE, MUSIC_INPUT_WIDTH};
use crate::resample::downsample_by_two;
use crate::{Error, Result};

/// Downsampled samples in one five-second detector window.
pub const MUSIC_WINDOW: usize = 55_125;
/// Downsampled samples between detector windows, equal to one source second.
pub const MUSIC_STEP: usize = 11_025;
const FFT_SIZE: usize = 512;
const FFT_HOP: usize = 256;

/// Reusable workspace for constructing music detector spectrograms.
#[derive(Debug, Clone)]
pub struct MusicFrontend {
    window: Vec<f32>,
    fft: GoogleFft,
    input: Vec<f32>,
    real: Vec<f32>,
    imag: Vec<f32>,
    work: Vec<f32>,
}

impl Default for MusicFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicFrontend {
    /// Allocates FFT scratch buffers and the periodic 512-point Hamming window.
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
            work: vec![0.0; FFT_SIZE],
        }
    }

    /// Converts one exact five-second downsampled window into detector input.
    ///
    /// `samples` must contain [`MUSIC_WINDOW`] float values at 11,025 Hz. The returned vector is
    /// frame-major with [`MUSIC_INPUT_SIZE`] linear magnitudes.
    pub fn process_f32(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() != MUSIC_WINDOW {
            return Err(Error::InvalidInput(format!(
                "music frontend needs {MUSIC_WINDOW} float samples and received {}",
                samples.len()
            )));
        }
        let mut output = Vec::with_capacity(MUSIC_INPUT_SIZE);
        for frame in 0..MUSIC_INPUT_HEIGHT {
            let start = frame * FFT_HOP;
            for index in 0..FFT_SIZE {
                self.input[index] = samples[start + index] * self.window[index];
            }
            self.real.fill(0.0);
            self.imag.fill(0.0);
            self.fft
                .transform_512(&self.input, &mut self.real, &mut self.imag, &mut self.work);
            output.push(self.real[0].abs());
            for bin in 1..MUSIC_INPUT_WIDTH - 1 {
                output.push(
                    (self.real[bin] * self.real[bin] + self.imag[bin] * self.imag[bin]).sqrt(),
                );
            }
            output.push(self.real[MUSIC_INPUT_WIDTH - 1].abs());
        }
        Ok(output)
    }

    /// Downsamples 22,050 Hz PCM16 and returns every complete detector window.
    ///
    /// Five-second windows advance by one second. Shorter inputs produce an empty vector and no
    /// partial detector score.
    pub fn process_stream(&mut self, samples: &[i16]) -> Result<Vec<Vec<f32>>> {
        let resampled = downsample_by_two(samples);
        self.process_resampled_stream(&resampled)
    }

    pub(crate) fn process_resampled_stream(&mut self, resampled: &[f32]) -> Result<Vec<Vec<f32>>> {
        if resampled.len() < MUSIC_WINDOW {
            return Ok(Vec::new());
        }
        let count = (resampled.len() - MUSIC_WINDOW) / MUSIC_STEP + 1;
        let mut outputs = Vec::with_capacity(count);
        for start in (0..=resampled.len() - MUSIC_WINDOW).step_by(MUSIC_STEP) {
            outputs.push(self.process_f32(&resampled[start..start + MUSIC_WINDOW])?);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_detector::MusicDetector;

    #[test]
    fn production_call_emits_four_inputs_from_eight_seconds() {
        let samples = vec![0; 22_050 * 8];
        assert_eq!(
            MusicFrontend::new().process_stream(&samples).unwrap().len(),
            4
        );
    }

    #[test]
    fn zero_pcm_matches_the_native_detector_score() {
        let inputs = MusicFrontend::new()
            .process_stream(&vec![0; 22_050 * 8])
            .unwrap();
        let detector = MusicDetector::from_library("../../assets/nnfp_v3.weights").unwrap();
        let expected = f32::from_bits(0x3b21_530a);
        for input in &inputs {
            let score = detector.infer(input).unwrap();
            assert!((score - expected).abs() <= 2.0e-9);
        }
    }
}
