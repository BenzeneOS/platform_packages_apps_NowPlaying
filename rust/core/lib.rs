//! Clean-room ambient music recognition using Google's model and index data.
//!
//! The production pipeline accepts event-rate PCM16, resamples it to 11,025 Hz, and runs a
//! five-second music detector before doing catalog work. Accepted audio is split into
//! overlapping two-second windows, converted to 42 by 512 magnitude spectra, and embedded as
//! 96 normalized floats. IVFADC search probes the local LevelDB shards for each embedding, then
//! the sequence scorer aligns candidates over time and accepts only scores strictly above the
//! configured threshold.
//!
//! Google's weights, quantizer configuration, and song shards are treated only as data. The
//! frontend, neural inference, table parsing, search, and scoring code are implemented here.
//! Correctness is measured against Google's recognizer on identical PCM. A filename is not ground
//! truth because a downloaded preview may not contain the indexed master recording.

/// Loads carved NNFP weights and produces normalized fingerprint embeddings.
pub mod embedder;
/// Converts PCM16 into the exact magnitude spectra consumed by the embedder.
pub mod frontend;
mod google_fft;
/// Parses matcher configuration and Google's standalone LevelDB song shards.
pub mod index;
/// Runs the recovered convolutional network used to reject non-music audio.
pub mod music_detector;
/// Builds the five-second magnitude spectra consumed by the music detector.
pub mod music_frontend;
/// Connects audio gating, embedding, sequence search, and WAV input.
pub mod recognize;
mod resample;
/// Computes the recovered aggregate sequence score and its adaptive inputs.
pub mod scorer;
/// Performs IVFADC lookup and time-aligned sequence search across installed shards.
pub mod search;
/// Reads the uncompressed standalone LevelDB table files used for matcher shards.
pub mod sstable;

use std::fmt;
use std::io;

/// An error raised while loading recognition data or processing a request.
#[derive(Debug)]
pub enum Error {
    /// A filesystem read failed while loading a weight, config, shard, or WAV file.
    Io(io::Error),
    /// Serialized model, protobuf, WAV, or LevelDB data is truncated or inconsistent.
    Format(String),
    /// Caller input has an unsupported rate, shape, identifier, or value.
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Format(message) | Self::InvalidInput(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// A result returned by the recognizer core.
pub type Result<T> = std::result::Result<T, Error>;
