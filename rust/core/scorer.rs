//! Aggregate scoring for a time-aligned embedding sequence.
//!
//! Per-query adaptive metrics are compared with aligned product-quantization distances. Repeated
//! query embeddings receive less weight, and background catalog mass contributes to the shared
//! bias. The final transform is
//!
//! ```text
//! z_i = metric_i - distance_scale * distance_i
//! u = sum_i weight_i * softplus(z_i)
//! score = bias + log(-expm1(-u)) + u
//! ```
//!
//! Larger scores are better. Production accepts only a score strictly greater than zero, never a
//! distance below zero or a score equal to zero.

use crate::index::{EMBEDDING_SIZE, ScorerConfig};
use crate::{Error, Result};

/// Measured combined background mass for the full production catalog.
pub const PRODUCTION_BACKGROUND_MASS: f32 = 18_476_420.0;

/// Query-specific state shared while scoring every candidate alignment.
///
/// Construction derives distinctiveness weights and the catalog-mass bias once. Each candidate
/// then supplies only one aligned asymmetric distance per query embedding.
#[derive(Debug, Clone)]
pub struct SequenceScorer {
    bias: f32,
    distance_scale: f32,
    weights: Vec<f32>,
    similarity_metrics: Vec<f32>,
}

impl SequenceScorer {
    /// Derives query weights and bias from embeddings, adaptive metrics, and background mass.
    ///
    /// Embeddings and metrics must have equal lengths. Background mass must be finite and
    /// positive because its logarithm contributes directly to the bias.
    pub fn new(
        config: ScorerConfig,
        embeddings: &[[f32; EMBEDDING_SIZE]],
        similarity_metrics: Vec<f32>,
        background_mass: f32,
    ) -> Result<Self> {
        if embeddings.len() != similarity_metrics.len() {
            return Err(Error::InvalidInput(
                "scorer inputs have different lengths".into(),
            ));
        }
        if !background_mass.is_finite() || background_mass <= 0.0 {
            return Err(Error::InvalidInput(
                "scorer background mass must be finite and positive".into(),
            ));
        }
        let weights = distinctiveness_weights(config, embeddings);
        let weight_sum = weights.iter().copied().sum::<f32>();
        let shape = softplus(config.weight_sum_shape);
        let normalization = if shape == 0.0 {
            weight_sum.ln()
        } else {
            (weight_sum - 1.0) * shape + ((-shape * weight_sum).exp_m1() / (-shape).exp_m1()).ln()
        };
        let bias = config.score_offset - background_mass.ln() - normalization;
        Ok(Self {
            bias,
            distance_scale: config.distance_scale,
            weights,
            similarity_metrics,
        })
    }

    /// Computes the recovered aggregate score for one aligned candidate sequence.
    ///
    /// `distances` must contain one value per query metric in the same temporal order. An empty
    /// sequence returns negative infinity. Callers accept only values strictly above their
    /// configured threshold.
    pub fn score(&self, distances: &[f32]) -> Result<f32> {
        if distances.len() != self.similarity_metrics.len() {
            return Err(Error::InvalidInput(
                "score distance count does not match the query".into(),
            ));
        }
        if distances.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }
        self.score_iter(distances.iter().copied())
    }

    pub(crate) fn score_iter(&self, distances: impl IntoIterator<Item = f32>) -> Result<f32> {
        let mut distances = distances.into_iter();
        let mut sum = 0.0f32;
        for (&metric, &weight) in self.similarity_metrics.iter().zip(&self.weights) {
            let Some(distance) = distances.next() else {
                return Err(Error::InvalidInput(
                    "score distance count does not match the query".into(),
                ));
            };
            let transformed = softplus(metric - distance * self.distance_scale);
            sum += weight * transformed;
        }
        if distances.next().is_some() {
            return Err(Error::InvalidInput(
                "score distance count does not match the query".into(),
            ));
        }
        Ok((-(-sum).exp_m1()).ln() + sum + self.bias)
    }

    /// Returns the distinctiveness weight for each query embedding.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns the adaptive similarity metric for each query embedding.
    pub fn similarity_metrics(&self) -> &[f32] {
        &self.similarity_metrics
    }

    /// Returns the catalog-mass and query-weight bias shared by candidate scores.
    pub fn bias(&self) -> f32 {
        self.bias
    }
}

/// Converts a retained neighbor distribution into one query similarity metric.
///
/// `minimum_distance` is the query's self-quantization distance. The furthest retained neighbor
/// and retained count adapt that baseline using the configured background mass. An empty neighbor
/// set leaves the baseline unchanged.
pub fn similarity_metric(
    config: ScorerConfig,
    minimum_distance: f32,
    nearest_distances: &[f32],
    background_mass: f32,
) -> f32 {
    let threshold = if nearest_distances.is_empty() {
        minimum_distance
    } else {
        let factor = (config.neighbor_mass_scale * background_mass
            / nearest_distances.len() as f32)
            .powf(config.neighbor_exponent.recip());
        let furthest = nearest_distances
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        minimum_distance + (furthest - minimum_distance) * factor
    };
    config.similarity_bias + threshold * config.similarity_scale
}

fn distinctiveness_weights(config: ScorerConfig, embeddings: &[[f32; EMBEDDING_SIZE]]) -> Vec<f32> {
    let mut row_sums = vec![1.0f32; embeddings.len()];
    for index in 0..embeddings.len() {
        for previous in 0..index {
            let mut squared_distance = 0.0f32;
            for (&current, &prior) in embeddings[index].iter().zip(&embeddings[previous]) {
                let difference = current - prior;
                squared_distance += difference * difference;
            }
            let contribution = 1.0
                / ((squared_distance * config.distinctiveness_scale - config.distinctiveness_bias)
                    .exp()
                    + 1.0);
            row_sums[index] += contribution;
            row_sums[previous] += contribution;
        }
    }
    row_sums.into_iter().map(f32::recip).collect()
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_native_winning_score_trace() {
        let scorer = SequenceScorer {
            bias: f32::from_bits(0xc17c_6de6),
            distance_scale: f32::from_bits(0x419a_f3a3),
            weights: [
                0x3ee9_3c0c,
                0x3ec9_2721,
                0x3ec0_b023,
                0x3ecd_ef07,
                0x3ecc_9c6e,
                0x3eec_f438,
                0x3f01_209d,
            ]
            .map(f32::from_bits)
            .to_vec(),
            similarity_metrics: [
                0x41e9_a723,
                0x41e7_1eff,
                0x41e8_0366,
                0x41e9_3082,
                0x41eb_8df3,
                0x41eb_4039,
                0x41e8_3465,
            ]
            .map(f32::from_bits)
            .to_vec(),
        };
        let distances = [
            0x3f98_4828,
            0x3f91_7270,
            0x3fb0_4488,
            0x3f9f_6d25,
            0x3faa_9fbb,
            0x3fa8_28d3,
            0x3f85_cf35,
        ]
        .map(f32::from_bits);
        let actual = scorer.score(&distances).unwrap();
        assert!((actual - 0.398_967_74).abs() < 2.0e-6);
    }
}
