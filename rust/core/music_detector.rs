//! Neural music gate used before fingerprint search.
//!
//! The detector consumes a five-second `214 * 257` magnitude spectrogram. It applies the
//! recovered log and global variance normalization, seven convolutions, and three dense layers
//! to produce one sigmoid score. Recognition proceeds when any complete detector window scores
//! above the policy threshold.

use std::fs;
use std::path::Path;

use crate::{Error, Result};

/// Spectrogram frames in one five-second detector input.
pub const MUSIC_INPUT_HEIGHT: usize = 214;
/// Magnitude bins retained per detector frame, including DC and Nyquist.
pub const MUSIC_INPUT_WIDTH: usize = 257;
/// Float count expected by the music detector network.
pub const MUSIC_INPUT_SIZE: usize = MUSIC_INPUT_HEIGHT * MUSIC_INPUT_WIDTH;

#[derive(Debug, Clone)]
struct Tensor {
    height: usize,
    width: usize,
    channels: usize,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ConvLayer {
    kernel: Vec<f32>,
    bias: Vec<f32>,
    kernel_height: usize,
    kernel_width: usize,
    input_channels: usize,
    output_channels: usize,
    pool: bool,
}

#[derive(Clone, Copy)]
struct ConvSpec {
    shape: (usize, usize, usize, usize),
    kernel: (usize, usize),
    bias: (usize, usize),
    pool: bool,
}

const CONVOLUTIONS: [ConvSpec; 7] = [
    ConvSpec {
        shape: (3, 3, 1, 8),
        kernel: (0x20c50, 0x20d70),
        bias: (0x20d70, 0x20d90),
        pool: true,
    },
    ConvSpec {
        shape: (3, 3, 8, 16),
        kernel: (0x20da0, 0x21fa0),
        bias: (0x21fa0, 0x21fe0),
        pool: true,
    },
    ConvSpec {
        shape: (3, 3, 16, 32),
        kernel: (0x21ff0, 0x267f0),
        bias: (0x267f0, 0x26870),
        pool: true,
    },
    ConvSpec {
        shape: (3, 3, 32, 64),
        kernel: (0x26880, 0x38880),
        bias: (0x38880, 0x38980),
        pool: true,
    },
    ConvSpec {
        shape: (3, 3, 64, 64),
        kernel: (0x38990, 0x5c990),
        bias: (0x5c990, 0x5ca90),
        pool: true,
    },
    ConvSpec {
        shape: (2, 2, 64, 32),
        kernel: (0x5caa0, 0x64aa0),
        bias: (0x64aa0, 0x64b20),
        pool: true,
    },
    ConvSpec {
        shape: (2, 2, 32, 32),
        kernel: (0x64b30, 0x68b30),
        bias: (0x68b30, 0x68bb0),
        pool: false,
    },
];

/// The recovered scalar music classifier loaded from native-layout weights.
///
/// The model has seven convolutional layers followed by dense layers of widths 8, 8, and 1.
/// Parameters occupy the native NNFP layout through offset `0x6bd14`.
#[derive(Debug, Clone)]
pub struct MusicDetector {
    convolutions: Vec<ConvLayer>,
    dense_0_kernel: Vec<f32>,
    dense_0_bias: Vec<f32>,
    dense_1_kernel: Vec<f32>,
    dense_1_bias: Vec<f32>,
    dense_2_kernel: Vec<f32>,
    dense_2_bias: f32,
}

impl MusicDetector {
    /// Loads the music detector tensors from a native-layout weight blob on disk.
    pub fn from_library(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_library_bytes(&fs::read(path)?)
    }

    /// Loads the music detector tensors from an in-memory native-layout weight blob.
    ///
    /// The data must retain the recovered offsets through `0x6bd14`. Truncated or misaligned
    /// tensor ranges return [`Error::Format`].
    pub fn from_library_bytes(library: &[u8]) -> Result<Self> {
        if library.len() < 0x6bd14 {
            return Err(Error::Format(format!(
                "NNFP library is too short at {} bytes",
                library.len()
            )));
        }
        let convolutions = CONVOLUTIONS
            .iter()
            .map(|spec| {
                let (kernel_height, kernel_width, input_channels, output_channels) = spec.shape;
                Ok(ConvLayer {
                    kernel: floats(library, spec.kernel)?,
                    bias: floats(library, spec.bias)?,
                    kernel_height,
                    kernel_width,
                    input_channels,
                    output_channels,
                    pool: spec.pool,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dense_2_bias = floats(library, (0x6bd10, 0x6bd14))?[0];
        Ok(Self {
            convolutions,
            dense_0_kernel: floats(library, (0x68bb0, 0x6bbb0))?,
            dense_0_bias: floats(library, (0x6bbb0, 0x6bbd0))?,
            dense_1_kernel: floats(library, (0x6bbd0, 0x6bcd0))?,
            dense_1_bias: floats(library, (0x6bcd0, 0x6bcf0))?,
            dense_2_kernel: floats(library, (0x6bcf0, 0x6bd10))?,
            dense_2_bias,
        })
    }

    /// Produces a sigmoid music score from one `214 * 257` magnitude spectrogram.
    ///
    /// Input values are transformed with `ln(x + 0.01)`, globally centered, and divided by the
    /// square root of population variance clamped to at least `0.01`. The result lies between zero
    /// and one, and thresholding is left to the recognition policy.
    pub fn infer(&self, input: &[f32]) -> Result<f32> {
        if input.len() != MUSIC_INPUT_SIZE {
            return Err(Error::InvalidInput(format!(
                "music detector needs {MUSIC_INPUT_SIZE} floats and received {}",
                input.len()
            )));
        }
        let mut tensor = Tensor {
            height: MUSIC_INPUT_HEIGHT,
            width: MUSIC_INPUT_WIDTH,
            channels: 1,
            values: normalize_log_input(input),
        };
        for layer in &self.convolutions {
            tensor = conv_same(&tensor, layer);
            add_bias_and_relu(&mut tensor, &layer.bias);
            if layer.pool {
                tensor = max_pool_2x2(&tensor);
            }
        }
        if (tensor.height, tensor.width, tensor.channels) != (3, 4, 32) {
            return Err(Error::Format(format!(
                "music detector head received tensor {} by {} by {}",
                tensor.height, tensor.width, tensor.channels
            )));
        }
        let hidden_0 = dense_relu(&tensor.values, &self.dense_0_kernel, &self.dense_0_bias);
        let hidden_1 = dense_relu(&hidden_0, &self.dense_1_kernel, &self.dense_1_bias);
        let mut logit = self.dense_2_bias;
        for (value, weight) in hidden_1.iter().zip(&self.dense_2_kernel) {
            logit = value.mul_add(*weight, logit);
        }
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

fn floats(data: &[u8], range: (usize, usize)) -> Result<Vec<f32>> {
    let bytes = data
        .get(range.0..range.1)
        .ok_or_else(|| Error::Format("music detector weight range exceeds the library".into()))?;
    if bytes.len() % 4 != 0 {
        return Err(Error::Format(
            "music detector weight range is not float32 aligned".into(),
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn normalize_log_input(input: &[f32]) -> Vec<f32> {
    let mut values = input
        .iter()
        .map(|value| (*value + 0.01).ln())
        .collect::<Vec<_>>();
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let difference = *value - mean;
            difference * difference
        })
        .sum::<f32>()
        / values.len() as f32;
    let inverse_stddev = variance.max(0.01).sqrt().recip();
    for value in &mut values {
        *value = (*value - mean) * inverse_stddev;
    }
    values
}

fn conv_same(input: &Tensor, layer: &ConvLayer) -> Tensor {
    assert_eq!(input.channels, layer.input_channels);
    let padding_top = (layer.kernel_height - 1) / 2;
    let padding_left = (layer.kernel_width - 1) / 2;
    let mut output = Tensor {
        height: input.height,
        width: input.width,
        channels: layer.output_channels,
        values: vec![0.0; input.height * input.width * layer.output_channels],
    };
    for output_y in 0..input.height {
        for output_x in 0..input.width {
            let output_start = (output_y * input.width + output_x) * layer.output_channels;
            for kernel_y in 0..layer.kernel_height {
                let input_y = output_y + kernel_y;
                if input_y < padding_top || input_y - padding_top >= input.height {
                    continue;
                }
                let input_y = input_y - padding_top;
                for kernel_x in 0..layer.kernel_width {
                    let input_x = output_x + kernel_x;
                    if input_x < padding_left || input_x - padding_left >= input.width {
                        continue;
                    }
                    let input_x = input_x - padding_left;
                    let input_start = (input_y * input.width + input_x) * input.channels;
                    for input_channel in 0..input.channels {
                        let input_value = input.values[input_start + input_channel];
                        let kernel_start = ((kernel_y * layer.kernel_width + kernel_x)
                            * input.channels
                            + input_channel)
                            * layer.output_channels;
                        accumulate_channels(
                            &mut output.values[output_start..output_start + layer.output_channels],
                            &layer.kernel[kernel_start..kernel_start + layer.output_channels],
                            input_value,
                        );
                    }
                }
            }
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_code)]
fn accumulate_channels(output: &mut [f32], kernel: &[f32], input: f32) {
    use std::arch::aarch64::{vfmaq_n_f32, vld1q_f32, vst1q_f32};

    assert_eq!(output.len(), kernel.len());
    let mut index = 0;
    while index + 4 <= output.len() {
        // SAFETY: equal slice lengths and the loop condition guarantee four initialized elements
        // at both pointers. `output` is exclusively borrowed and the intrinsic keeps no pointer.
        unsafe {
            let values = vld1q_f32(output.as_ptr().add(index));
            let weights = vld1q_f32(kernel.as_ptr().add(index));
            vst1q_f32(
                output.as_mut_ptr().add(index),
                vfmaq_n_f32(values, weights, input),
            );
        }
        index += 4;
    }
    for (value, &weight) in output[index..].iter_mut().zip(&kernel[index..]) {
        *value = input.mul_add(weight, *value);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn accumulate_channels(output: &mut [f32], kernel: &[f32], input: f32) {
    for (value, &weight) in output.iter_mut().zip(kernel) {
        *value = input.mul_add(weight, *value);
    }
}

fn add_bias_and_relu(tensor: &mut Tensor, bias: &[f32]) {
    assert_eq!(tensor.channels, bias.len());
    for values in tensor.values.chunks_exact_mut(tensor.channels) {
        for (value, bias) in values.iter_mut().zip(bias) {
            *value = (*value + bias).max(0.0);
        }
    }
}

fn max_pool_2x2(input: &Tensor) -> Tensor {
    let output_height = input.height / 2;
    let output_width = input.width / 2;
    let mut output = Tensor {
        height: output_height,
        width: output_width,
        channels: input.channels,
        values: vec![0.0; output_height * output_width * input.channels],
    };
    for output_y in 0..output_height {
        for output_x in 0..output_width {
            let output_start = (output_y * output_width + output_x) * input.channels;
            for channel in 0..input.channels {
                let mut maximum = f32::NEG_INFINITY;
                for kernel_y in 0..2 {
                    for kernel_x in 0..2 {
                        let input_y = output_y * 2 + kernel_y;
                        let input_x = output_x * 2 + kernel_x;
                        let index = (input_y * input.width + input_x) * input.channels + channel;
                        maximum = maximum.max(input.values[index]);
                    }
                }
                output.values[output_start + channel] = maximum;
            }
        }
    }
    output
}

fn dense_relu(input: &[f32], kernel: &[f32], bias: &[f32]) -> Vec<f32> {
    assert_eq!(kernel.len(), input.len() * bias.len());
    let mut output = bias.to_vec();
    for (input_index, input_value) in input.iter().copied().enumerate() {
        for output_index in 0..output.len() {
            let index = input_index * output.len() + output_index;
            output[output_index] = input_value.mul_add(kernel[index], output[output_index]);
        }
    }
    for value in &mut output {
        *value = value.max(0.0);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_all_recovered_parameters() {
        let detector = MusicDetector::from_library("../../assets/nnfp_v3.weights").unwrap();
        assert_eq!(detector.convolutions.len(), 7);
        assert_eq!(detector.dense_0_kernel.len(), 384 * 8);
        assert_eq!(detector.dense_1_kernel.len(), 8 * 8);
        assert_eq!(detector.dense_2_kernel.len(), 8);
    }
}
