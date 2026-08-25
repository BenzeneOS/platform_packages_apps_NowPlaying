//! NNFP embedding inference over carved Google model weights.
//!
//! The weight blob preserves the native library's tensor offsets and contains 35 little-endian
//! float32 tensors. The graph itself is implemented here as ten convolutions with grouped
//! spatial normalization and ELU, followed by the recovered two-stage head. Callers only supply
//! the frontend's flat `42 * 512` magnitude tensor.

use std::fs;
use std::path::Path;

use crate::frontend::MODEL_INPUT_SIZE;
use crate::{Error, Result};

/// Float count in each L2-normalized fingerprint embedding.
pub const EMBEDDING_SIZE: usize = 96;
const INPUT_EPSILON: f32 = 1.0e-8;
const SPATIAL_EPSILON: f32 = 0.1;
const OUTPUT_NORM_FLOOR: f32 = 1.0e-12;
const NORMALIZATION_GROUPS: usize = 16;

#[derive(Debug, Clone)]
struct Layer {
    kernel: Vec<f32>,
    scale: Vec<f32>,
    bias: Vec<f32>,
    kernel_height: usize,
    kernel_width: usize,
    input_channels: usize,
    output_channels: usize,
    stride_height: usize,
    stride_width: usize,
}

/// The clean-room NNFP network loaded from a native-layout weight blob.
///
/// The blob supplies parameters only. Convolution order, normalization, activation, the head's
/// scratch layout, and output normalization are implemented by this type.
#[derive(Debug, Clone)]
pub struct Embedder {
    layers: Vec<Layer>,
    head_kernel: Vec<f32>,
    head_scale: Vec<f32>,
    head_bias: Vec<f32>,
    output_kernel: Vec<f32>,
    output_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Tensor {
    height: usize,
    width: usize,
    channels: usize,
    values: Vec<f32>,
}

#[derive(Clone, Copy)]
struct LayerSpec {
    kernel: (usize, usize),
    scale: (usize, usize),
    bias: (usize, usize),
    shape: (usize, usize, usize, usize),
    stride: (usize, usize),
}

const LAYERS: [LayerSpec; 10] = [
    LayerSpec {
        kernel: (0x6c8ec, 0x6c9ec),
        scale: (0x6c9ec, 0x6ca2c),
        bias: (0x6ca2c, 0x6ca6c),
        shape: (1, 4, 1, 16),
        stride: (1, 2),
    },
    LayerSpec {
        kernel: (0x6ca7c, 0x6ea7c),
        scale: (0x6ea7c, 0x6eafc),
        bias: (0x6eafc, 0x6eb7c),
        shape: (4, 1, 16, 32),
        stride: (2, 1),
    },
    LayerSpec {
        kernel: (0x6eb8c, 0x74b8c),
        scale: (0x74b8c, 0x74c4c),
        bias: (0x74c4c, 0x74d0c),
        shape: (1, 4, 32, 48),
        stride: (1, 2),
    },
    LayerSpec {
        kernel: (0x74d1c, 0x80d1c),
        scale: (0x80d1c, 0x80e1c),
        bias: (0x80e1c, 0x80f1c),
        shape: (4, 1, 48, 64),
        stride: (2, 1),
    },
    LayerSpec {
        kernel: (0x80f2c, 0x98f2c),
        scale: (0x98f2c, 0x990ac),
        bias: (0x990ac, 0x9922c),
        shape: (1, 4, 64, 96),
        stride: (1, 2),
    },
    LayerSpec {
        kernel: (0x9923c, 0xc923c),
        scale: (0xc923c, 0xc943c),
        bias: (0xc943c, 0xc963c),
        shape: (4, 1, 96, 128),
        stride: (2, 1),
    },
    LayerSpec {
        kernel: (0xc964c, 0x12964c),
        scale: (0x12964c, 0x12994c),
        bias: (0x12994c, 0x129c4c),
        shape: (1, 4, 128, 192),
        stride: (1, 2),
    },
    LayerSpec {
        kernel: (0x129c5c, 0x1e9c5c),
        scale: (0x1e9c5c, 0x1ea05c),
        bias: (0x1ea05c, 0x1ea45c),
        shape: (4, 1, 192, 256),
        stride: (2, 1),
    },
    LayerSpec {
        kernel: (0x1ea46c, 0x2aa46c),
        scale: (0x2aa46c, 0x2aa76c),
        bias: (0x2aa76c, 0x2aaa6c),
        shape: (1, 4, 256, 192),
        stride: (1, 1),
    },
    LayerSpec {
        kernel: (0x2aaa7c, 0x30aa7c),
        scale: (0x30aa7c, 0x30ac7c),
        bias: (0x30ac7c, 0x30ae7c),
        shape: (4, 1, 192, 128),
        stride: (1, 1),
    },
];

impl Embedder {
    /// Loads all 35 tensors from a native-layout weight blob on disk.
    ///
    /// Both the zero-padded carved asset and the original NNFP shared library have the offsets
    /// expected by this parser.
    pub fn from_library(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_library_bytes(&fs::read(path)?)
    }

    /// Loads all 35 tensors from an in-memory native-layout weight blob.
    ///
    /// The data must extend through native offset `0x3cf7fc`. Tensor ranges are fixed by the
    /// recovered NNFP v3 layout and malformed or truncated input returns [`Error::Format`].
    pub fn from_library_bytes(library: &[u8]) -> Result<Self> {
        if library.len() < 0x3cf7fc {
            return Err(Error::Format(format!(
                "NNFP library is too short at {} bytes",
                library.len()
            )));
        }

        let layers = LAYERS
            .iter()
            .map(|spec| {
                let (kernel_height, kernel_width, input_channels, output_channels) = spec.shape;
                Ok(Layer {
                    kernel: floats(library, spec.kernel)?,
                    scale: floats(library, spec.scale)?,
                    bias: floats(library, spec.bias)?,
                    kernel_height,
                    kernel_width,
                    input_channels,
                    output_channels,
                    stride_height: spec.stride.0,
                    stride_width: spec.stride.1,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            layers,
            head_kernel: floats(library, (0x30ae7c, 0x3cae7c))?,
            head_scale: floats(library, (0x3cae7c, 0x3cc67c))?,
            head_bias: floats(library, (0x3cc67c, 0x3cde7c))?,
            output_kernel: floats(library, (0x3cde7c, 0x3cf67c))?,
            output_bias: floats(library, (0x3cf67c, 0x3cf7fc))?,
        })
    }

    /// Converts one frontend tensor into a normalized 96-dimensional fingerprint.
    ///
    /// `input` must contain exactly [`MODEL_INPUT_SIZE`] frame-major magnitudes. The returned
    /// vector is L2-normalized and can be passed directly to the IVFADC search functions.
    pub fn infer(&self, input: &[f32]) -> Result<[f32; EMBEDDING_SIZE]> {
        if input.len() != MODEL_INPUT_SIZE {
            return Err(Error::InvalidInput(format!(
                "embedder needs {MODEL_INPUT_SIZE} floats and received {}",
                input.len()
            )));
        }

        let mut tensor = Tensor {
            height: 42,
            width: 512,
            channels: 1,
            values: normalize_input(input),
        };
        for layer in &self.layers {
            tensor = conv_same(&tensor, layer);
            normalize_affine(&mut tensor, &layer.scale, &layer.bias);
            elu(&mut tensor.values);
        }
        self.head(&tensor)
    }

    fn head(&self, tensor: &Tensor) -> Result<[f32; EMBEDDING_SIZE]> {
        if (tensor.height, tensor.width, tensor.channels) != (3, 32, 128) {
            return Err(Error::Format(format!(
                "head received tensor {} by {} by {}",
                tensor.height, tensor.width, tensor.channels
            )));
        }

        let mut hidden = vec![0.0f32; 16 * EMBEDDING_SIZE];
        for hidden_channel in 0..16 {
            for output_dimension in 0..EMBEDDING_SIZE {
                let mut sum = 0.0f32;
                for input_channel in 0..128 {
                    let input_index = input_channel * EMBEDDING_SIZE + output_dimension;
                    let kernel_index =
                        (input_channel * EMBEDDING_SIZE + output_dimension) * 16 + hidden_channel;
                    sum += tensor.values[input_index] * self.head_kernel[kernel_index];
                }
                hidden[output_dimension * 16 + hidden_channel] = sum;
            }
        }

        for hidden_channel in 0..16 {
            let range = hidden_channel * EMBEDDING_SIZE..(hidden_channel + 1) * EMBEDDING_SIZE;
            let values = &hidden[range.clone()];
            let mean = values.iter().copied().sum::<f32>() / EMBEDDING_SIZE as f32;
            let variance = values
                .iter()
                .map(|value| {
                    let difference = *value - mean;
                    difference * difference
                })
                .sum::<f32>()
                / EMBEDDING_SIZE as f32;
            let inverse_stddev = (variance + SPATIAL_EPSILON).sqrt().recip();
            for index in range {
                let factor = self.head_scale[index] * inverse_stddev;
                let offset = self.head_bias[index] - mean * factor;
                hidden[index] = hidden[index] * factor + offset;
            }
        }
        elu(&mut hidden);

        let mut output = [0.0f32; EMBEDDING_SIZE];
        for (output_dimension, output_value) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for hidden_channel in 0..16 {
                let index = hidden_channel * EMBEDDING_SIZE + output_dimension;
                sum += hidden[index] * self.output_kernel[index];
            }
            *output_value = sum + self.output_bias[output_dimension];
        }

        let squared_norm = output
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .max(OUTPUT_NORM_FLOOR);
        let inverse_norm = squared_norm.sqrt().recip();
        for value in &mut output {
            *value *= inverse_norm;
        }
        Ok(output)
    }
}

fn floats(data: &[u8], range: (usize, usize)) -> Result<Vec<f32>> {
    let bytes = data
        .get(range.0..range.1)
        .ok_or_else(|| Error::Format("weight range extends beyond the library".into()))?;
    if bytes.len() % 4 != 0 {
        return Err(Error::Format("weight range is not float32 aligned".into()));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn normalize_input(input: &[f32]) -> Vec<f32> {
    let mean = input.iter().copied().sum::<f32>() / input.len() as f32;
    let variance = input
        .iter()
        .map(|value| {
            let difference = *value - mean;
            difference * difference
        })
        .sum::<f32>()
        / input.len() as f32;
    let inverse_stddev = (variance + INPUT_EPSILON).sqrt().recip();
    input
        .iter()
        .map(|value| (*value - mean) * inverse_stddev)
        .collect()
}

fn conv_same(input: &Tensor, layer: &Layer) -> Tensor {
    assert_eq!(input.channels, layer.input_channels);
    let output_height = input.height.div_ceil(layer.stride_height);
    let output_width = input.width.div_ceil(layer.stride_width);
    let padding_height = ((output_height - 1) * layer.stride_height + layer.kernel_height)
        .saturating_sub(input.height);
    let padding_width =
        ((output_width - 1) * layer.stride_width + layer.kernel_width).saturating_sub(input.width);
    let padding_top = padding_height / 2;
    let padding_left = padding_width / 2;
    let mut output = Tensor {
        height: output_height,
        width: output_width,
        channels: layer.output_channels,
        values: vec![0.0; output_height * output_width * layer.output_channels],
    };

    for output_y in 0..output_height {
        for output_x in 0..output_width {
            let output_start = (output_y * output_width + output_x) * layer.output_channels;
            for kernel_y in 0..layer.kernel_height {
                let input_y = output_y * layer.stride_height + kernel_y;
                if input_y < padding_top || input_y - padding_top >= input.height {
                    continue;
                }
                let input_y = input_y - padding_top;
                for kernel_x in 0..layer.kernel_width {
                    let input_x = output_x * layer.stride_width + kernel_x;
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

fn normalize_affine(tensor: &mut Tensor, scale: &[f32], bias: &[f32]) {
    assert_eq!(scale.len(), tensor.channels);
    assert_eq!(bias.len(), tensor.channels);
    assert_eq!(tensor.channels % NORMALIZATION_GROUPS, 0);
    let spatial_size = tensor.height * tensor.width;
    let group_width = tensor.channels / NORMALIZATION_GROUPS;
    for group in 0..NORMALIZATION_GROUPS {
        let channel_start = group * group_width;
        let channel_end = channel_start + group_width;
        let mut sum = 0.0f32;
        for spatial in 0..spatial_size {
            let spatial_start = spatial * tensor.channels;
            for channel in channel_start..channel_end {
                sum += tensor.values[spatial_start + channel];
            }
        }
        let element_count = spatial_size * group_width;
        let mean = sum / element_count as f32;
        let mut squared_sum = 0.0f32;
        for spatial in 0..spatial_size {
            let spatial_start = spatial * tensor.channels;
            for channel in channel_start..channel_end {
                let difference = tensor.values[spatial_start + channel] - mean;
                squared_sum += difference * difference;
            }
        }
        let inverse_stddev = (squared_sum / element_count as f32 + SPATIAL_EPSILON)
            .sqrt()
            .recip();
        for spatial in 0..spatial_size {
            let spatial_start = spatial * tensor.channels;
            for channel in channel_start..channel_end {
                let factor = scale[channel] * inverse_stddev;
                let offset = bias[channel] - mean * factor;
                let value = &mut tensor.values[spatial_start + channel];
                *value = *value * factor + offset;
            }
        }
    }
}

fn elu(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = value.exp() - 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    #[test]
    fn loads_all_recovered_tensors() {
        let embedder = Embedder::from_library("../../assets/nnfp_v3.weights").unwrap();
        assert_eq!(embedder.layers.len(), 10);
        assert_eq!(embedder.head_kernel.len(), 128 * 96 * 16);
        assert_eq!(embedder.head_scale.len(), 16 * 96);
        assert_eq!(embedder.head_bias.len(), 16 * 96);
        assert_eq!(embedder.output_kernel.len(), 16 * 96);
        assert_eq!(embedder.output_bias.len(), 96);
    }

    #[test]
    fn matches_google_golden_embedding_with_host_rounding_bound() {
        const MAX_ABSOLUTE_ERROR: f32 = 6.0e-7;
        const MIN_COSINE: f64 = 0.999_999_999;

        let input = fixture_floats(include_bytes!("../tests/fixtures/golden-input.f32le"));
        let expected = fixture_floats(include_bytes!("../tests/fixtures/golden-output.f32le"));
        let embedder = Embedder::from_library("../../assets/nnfp_v3.weights").unwrap();
        let actual = embedder.infer(&input).unwrap();
        let max_absolute_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let dot = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| f64::from(*actual) * f64::from(*expected))
            .sum::<f64>();
        let actual_norm = actual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected_norm = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = dot / (actual_norm * expected_norm);
        assert!(max_absolute_error <= MAX_ABSOLUTE_ERROR);
        assert!(cosine >= MIN_COSINE);
    }
}
