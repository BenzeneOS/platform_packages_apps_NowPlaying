#![allow(missing_docs)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nowplaying_core::embedder::Embedder;
use nowplaying_core::index::{MatcherConfig, Shard, ShardSet, TreeIdDecoder};
use nowplaying_core::music_detector::MusicDetector;
use nowplaying_core::recognize::{RecognitionPolicy, Recognizer};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let separator = arguments
        .iter()
        .position(|argument| argument == "--")
        .ok_or("missing capture separator")?;
    let setup = &arguments[..separator];
    let captures = &arguments[separator + 1..];
    if setup.len() < 3 || captures.is_empty() {
        return Err("usage WEIGHTS CONFIG SHARD... -- CAPTURE...".into());
    }
    let weights = &setup[0];
    let config = MatcherConfig::from_file(&setup[1])?;
    let decoder = Arc::new(TreeIdDecoder::from_library(weights)?);
    let shards = setup[2..]
        .iter()
        .map(|path| {
            let name = Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("shard path has no UTF-8 file name")?;
            Shard::open(name, path, Arc::clone(&decoder)).map_err(|error| error.into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut recognizer = Recognizer::new(
        Embedder::from_library(weights)?,
        MusicDetector::from_library(weights)?,
        ShardSet::new(config, shards),
        RecognitionPolicy::default(),
    )?;
    println!("capture\taccepted\ttrack\tnumeric_id\tshard\toffset\tscore\ttotal_us");
    for capture in captures {
        let (samples, sample_rate) = read_wav(capture)?;
        let outcome = recognizer.recognize_pcm_timed(&samples, sample_rate)?;
        match outcome.recognition {
            Some(recognition) => {
                let result = recognition.result;
                println!(
                    "{}\t1\t{}\t{}\t{}\t{}\t{}\t{}",
                    capture,
                    result.metadata.track_id,
                    result.track_id,
                    result.shard_name,
                    result.offset_seconds,
                    result.score,
                    outcome.timings.total.as_micros(),
                );
            }
            None => println!(
                "{}\t0\t\t\t\t\t\t{}",
                capture,
                outcome.timings.total.as_micros(),
            ),
        }
    }
    Ok(())
}

fn read_wav(path: impl Into<PathBuf>) -> Result<(Vec<i16>, u32), Box<dyn Error>> {
    let path = path.into();
    let data = std::fs::read(path)?;
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("capture is not a RIFF WAVE file".into());
    }
    let mut offset = 12;
    let mut format = None;
    let mut pcm = None;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        offset += 8;
        let end = offset.checked_add(chunk_size).ok_or("WAV chunk overflow")?;
        let chunk = data.get(offset..end).ok_or("truncated WAV chunk")?;
        if chunk_id == b"fmt " && chunk.len() >= 16 {
            format = Some((
                u16::from_le_bytes(chunk[0..2].try_into()?),
                u16::from_le_bytes(chunk[2..4].try_into()?),
                u32::from_le_bytes(chunk[4..8].try_into()?),
                u16::from_le_bytes(chunk[14..16].try_into()?),
            ));
        } else if chunk_id == b"data" {
            pcm = Some(chunk);
        }
        offset = end + (chunk_size & 1);
    }
    let (encoding, channels, sample_rate, bits_per_sample) = format.ok_or("missing WAV format")?;
    if encoding != 1 || channels != 1 || bits_per_sample != 16 {
        return Err("capture must contain mono PCM16".into());
    }
    let pcm = pcm.ok_or("missing WAV PCM")?;
    if pcm.len() % 2 != 0 {
        return Err("WAV PCM ends inside a sample".into());
    }
    let samples = pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| i16::from_le_bytes(*bytes))
        .collect();
    Ok((samples, sample_rate))
}
