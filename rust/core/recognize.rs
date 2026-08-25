//! High-level ambient music recognition pipeline.
//!
//! [`crate::recognize::Recognizer::recognize_pcm`] first resamples PCM16 to 11,025 Hz and evaluates
//! overlapping five-second detector windows. If any detector score is strictly above the policy
//! threshold, the same stream is converted into overlapping two-second fingerprint
//! inputs, embedded, searched across every shard, and aligned by the sequence scorer. The first
//! ranked result strictly above the matcher acceptance threshold is returned.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::embedder::Embedder;
use crate::frontend::{Frontend, SAMPLE_RATE};
use crate::index::ShardSet;
use crate::music_detector::MusicDetector;
use crate::music_frontend::MusicFrontend;
use crate::resample::resample;
use crate::search::{
    ContinuityScore, PreviousMatch, SequenceSearch, SequenceSearchResult, search_sequences,
    search_sequences_timed_with_previous,
};
use crate::{Error, Result};

/// Runtime gates controlling high-level recognition work.
#[derive(Debug, Clone, Copy)]
pub struct RecognitionPolicy {
    /// Maximum sequence candidates retained after aggregate scoring.
    pub candidate_limit: usize,
    /// Strict lower bound that any detector window must exceed before shard search runs.
    pub music_threshold: f32,
}

impl Default for RecognitionPolicy {
    fn default() -> Self {
        Self {
            candidate_limit: 10,
            music_threshold: 0.2,
        }
    }
}

/// One accepted recognition result.
#[derive(Debug, Clone)]
pub struct Recognition {
    /// Highest-ranked sequence result above the matcher acceptance threshold.
    pub result: SequenceSearchResult,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RecognitionTimings {
    pub resample: Duration,
    pub music_gate: Duration,
    pub frontend: Duration,
    pub embed: Duration,
    pub search: Duration,
    pub score: Duration,
    pub total: Duration,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct RecognitionOutcome {
    pub recognition: Option<Recognition>,
    pub music_score: Option<f32>,
    pub continuity: Option<ContinuityScore>,
    pub timings: RecognitionTimings,
}

/// Stateful owner of the detector, embedder, shard set, and reusable frontend buffers.
///
/// A recognizer can process repeated PCM requests without rebuilding model state. It does not
/// retain audio or cross-call voting state.
pub struct Recognizer {
    frontend: Frontend,
    embedder: Embedder,
    music_frontend: MusicFrontend,
    music_detector: MusicDetector,
    shards: ShardSet,
    policy: RecognitionPolicy,
}

impl Recognizer {
    /// Constructs a recognizer from loaded models, shards, and runtime policy.
    ///
    /// The candidate limit must be positive. Model and shard compatibility is checked when their
    /// individual loaders parse the supplied data.
    pub fn new(
        embedder: Embedder,
        music_detector: MusicDetector,
        shards: ShardSet,
        policy: RecognitionPolicy,
    ) -> Result<Self> {
        if policy.candidate_limit == 0 {
            return Err(Error::InvalidInput(
                "recognition candidate count must be positive".into(),
            ));
        }
        Ok(Self {
            frontend: Frontend::new(),
            embedder,
            music_frontend: MusicFrontend::new(),
            music_detector,
            shards,
            policy,
        })
    }

    #[allow(missing_docs)]
    pub fn replace_shards(&mut self, shards: ShardSet) {
        self.shards = shards;
    }

    /// Runs the music gate and full recognition pipeline over PCM16 at the supplied rate.
    ///
    /// Any detector window must score strictly above [`RecognitionPolicy::music_threshold`]. After
    /// search, a sequence score must be strictly above
    /// [`crate::index::MatcherConfig::acceptance_threshold`].
    /// Audio too short to produce a detector window returns `Ok(None)`.
    pub fn recognize_pcm(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<Option<Recognition>> {
        Ok(self.recognize_pcm_timed(samples, sample_rate)?.recognition)
    }

    #[allow(missing_docs)]
    pub fn recognize_pcm_timed(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<RecognitionOutcome> {
        self.recognize_pcm_timed_with_context(samples, sample_rate, true, None)
    }

    #[allow(missing_docs)]
    pub fn recognize_pcm_timed_with_context(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
        fingerprint_matching_enabled: bool,
        previous_match: Option<&PreviousMatch>,
    ) -> Result<RecognitionOutcome> {
        let total_started = Instant::now();
        let phase_started = Instant::now();
        let resampled = resample(samples, sample_rate)?;
        let resample = phase_started.elapsed();
        let phase_started = Instant::now();
        let music_inputs = self.music_frontend.process_resampled_stream(&resampled)?;
        let music_scores = parallel_map(&music_inputs, |input| self.music_detector.infer(input))?;
        let music_score = music_scores.first().copied();
        let music_gate = phase_started.elapsed();
        if !fingerprint_matching_enabled
            || !music_scores
                .into_iter()
                .any(|score| score > self.policy.music_threshold)
        {
            return Ok(RecognitionOutcome {
                recognition: None,
                music_score,
                continuity: None,
                timings: RecognitionTimings {
                    resample,
                    music_gate,
                    total: total_started.elapsed(),
                    ..RecognitionTimings::default()
                },
            });
        }
        let phase_started = Instant::now();
        let frontend = self.frontend.process_resampled_stream(&resampled)?;
        let frontend_time = phase_started.elapsed();
        let phase_started = Instant::now();
        let embeddings =
            parallel_map(&frontend, |output| self.embedder.infer(&output.model_input))?;
        let embed = phase_started.elapsed();
        let search = search_sequences_timed_with_previous(
            &self.shards,
            &embeddings,
            self.policy.candidate_limit,
            previous_match,
        );
        self.shards.discard_mapped_pages();
        let (search, search_timings) = search?;
        let continuity = search.continuity;
        let recognition = search
            .results
            .into_iter()
            .find(|result| result.score > self.shards.config.acceptance_threshold)
            .map(|result| Recognition { result });
        Ok(RecognitionOutcome {
            recognition,
            music_score,
            continuity,
            timings: RecognitionTimings {
                resample,
                music_gate,
                frontend: frontend_time,
                embed,
                search: search_timings.search,
                score: search_timings.score,
                total: total_started.elapsed(),
            },
        })
    }

    /// Returns one detector score for each complete five-second window in event-rate PCM16 input.
    ///
    /// Windows advance by one second after resampling to 11,025 Hz. No policy threshold is applied,
    /// which makes this useful for gate diagnostics.
    pub fn music_scores(&mut self, samples: &[i16], sample_rate: u32) -> Result<Vec<f32>> {
        let resampled = resample(samples, sample_rate)?;
        self.music_scores_resampled(&resampled)
    }

    fn music_scores_resampled(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        self.music_frontend
            .process_resampled_stream(samples)?
            .iter()
            .map(|input| self.music_detector.infer(input))
            .collect()
    }

    /// Embeds and searches event-rate PCM16 without applying the music gate or acceptance threshold.
    ///
    /// The returned diagnostics include rejected sequence scores. Fingerprint windows cover two
    /// source seconds and advance one second after resampling to 11,025 Hz.
    pub fn search_pcm(&mut self, samples: &[i16], sample_rate: u32) -> Result<SequenceSearch> {
        let resampled = resample(samples, sample_rate)?;
        self.search_resampled(&resampled)
    }

    fn search_resampled(&mut self, samples: &[f32]) -> Result<SequenceSearch> {
        let mut embeddings = Vec::new();
        for output in self.frontend.process_resampled_stream(samples)? {
            embeddings.push(self.embedder.infer(&output.model_input)?);
        }
        search_sequences(&self.shards, &embeddings, self.policy.candidate_limit)
    }

    /// Searches precomputed embeddings without running either audio frontend or music gate.
    ///
    /// This parity entry point still applies the matcher acceptance threshold and returns only the
    /// highest-ranked accepted result.
    pub fn recognize_embeddings(&self, embeddings: &[[f32; 96]]) -> Result<Option<Recognition>> {
        let search = search_sequences(&self.shards, embeddings, self.policy.candidate_limit)?;
        Ok(search
            .results
            .into_iter()
            .find(|result| result.score > self.shards.config.acceptance_threshold)
            .map(|result| Recognition { result }))
    }
}

fn parallel_map<T, U, F>(inputs: &[T], operation: F) -> Result<Vec<U>>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Result<U> + Sync,
{
    let worker_count = inputs.len().clamp(1, 4);
    let mut outputs = std::thread::scope(|scope| {
        let operation = &operation;
        let mut workers = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            workers.push(scope.spawn(move || {
                inputs
                    .iter()
                    .enumerate()
                    .skip(worker_index)
                    .step_by(worker_count)
                    .map(|(index, input)| operation(input).map(|output| (index, output)))
                    .collect::<Result<Vec<_>>>()
            }));
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| Error::Format("recognition worker panicked".into()))?
            })
            .collect::<Result<Vec<Vec<_>>>>()
    })?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    outputs.sort_unstable_by_key(|(index, _)| *index);
    Ok(outputs.into_iter().map(|(_, output)| output).collect())
}

/// Reads a RIFF WAVE file and decodes supported PCM16 samples.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Vec<i16>> {
    parse_wav(&fs::read(path)?)
}

/// Decodes mono or stereo 22,050 Hz PCM16 from a RIFF WAVE byte stream.
///
/// Stereo frames are averaged to mono with signed integer arithmetic. Other encodings, sample
/// rates, channel counts, truncated chunks, and partial sample frames are rejected.
pub fn parse_wav(data: &[u8]) -> Result<Vec<i16>> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(Error::Format("input is not a RIFF WAVE file".into()));
    }
    let mut offset = 12;
    let mut format = None;
    let mut pcm = None;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let end = offset
            .checked_add(chunk_size)
            .ok_or_else(|| Error::Format("WAV chunk range overflows usize".into()))?;
        let chunk = data
            .get(offset..end)
            .ok_or_else(|| Error::Format("WAV chunk is truncated".into()))?;
        if chunk_id == b"fmt " {
            if chunk.len() < 16 {
                return Err(Error::Format("WAV format chunk is truncated".into()));
            }
            format = Some((
                u16::from_le_bytes(chunk[0..2].try_into().unwrap()),
                u16::from_le_bytes(chunk[2..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                u16::from_le_bytes(chunk[14..16].try_into().unwrap()),
            ));
        } else if chunk_id == b"data" {
            pcm = Some(chunk);
        }
        offset = end + (chunk_size & 1);
    }
    let (encoding, channels, sample_rate, bits_per_sample) =
        format.ok_or_else(|| Error::Format("WAV has no format chunk".into()))?;
    if encoding != 1 || bits_per_sample != 16 {
        return Err(Error::InvalidInput("WAV must contain PCM16".into()));
    }
    if sample_rate != SAMPLE_RATE as u32 {
        return Err(Error::InvalidInput(format!(
            "WAV sample rate must be {SAMPLE_RATE} Hz"
        )));
    }
    if channels != 1 && channels != 2 {
        return Err(Error::InvalidInput("WAV must be mono or stereo".into()));
    }
    let bytes = pcm.ok_or_else(|| Error::Format("WAV has no data chunk".into()))?;
    if bytes.len() % (usize::from(channels) * 2) != 0 {
        return Err(Error::Format("WAV data ends inside a sample frame".into()));
    }
    let decoded = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| i16::from_le_bytes(*sample))
        .collect::<Vec<_>>();
    if channels == 1 {
        return Ok(decoded);
    }
    Ok(decoded
        .as_chunks::<2>()
        .0
        .iter()
        .map(|frame| ((i32::from(frame[0]) + i32::from(frame[1])) / 2) as i16)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono_wav(samples: &[i16]) -> Vec<u8> {
        let data_size = samples.len() * 2;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + data_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE as u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    #[test]
    fn reads_mono_pcm16() {
        assert_eq!(parse_wav(&mono_wav(&[-1, 2, 3])).unwrap(), [-1, 2, 3]);
    }

    #[test]
    fn rejects_an_unpinned_sample_rate() {
        let mut wav = mono_wav(&[0]);
        wav[24..28].copy_from_slice(&44_100u32.to_le_bytes());
        assert!(parse_wav(&wav).is_err());
    }

    #[test]
    #[ignore = "requires NOWPLAYING_ARTIFACTS_PATH and NOWPLAYING_16KHZ_WAV"]
    fn recognizes_the_device_16_khz_capture() {
        use std::sync::Arc;

        use crate::index::{MatcherConfig, Shard, ShardSet, TreeIdDecoder};

        let artifacts = std::env::var("NOWPLAYING_ARTIFACTS_PATH").unwrap();
        let wav_path = std::env::var("NOWPLAYING_16KHZ_WAV").unwrap();
        let decoder =
            Arc::new(TreeIdDecoder::from_library("../../assets/nnfp_v3.weights").unwrap());
        let shards = ShardSet::new(
            MatcherConfig::from_file(format!("{artifacts}/etc/ambient/v3_config_tah.pb")).unwrap(),
            vec![
                Shard::open(
                    "matcher_tah.leveldb",
                    format!("{artifacts}/etc/ambient/matcher_tah.leveldb"),
                    decoder.clone(),
                )
                .unwrap(),
                Shard::open(
                    "US0d8db236a287d657caca26073f6165d8",
                    format!("{artifacts}/etc/ambient/US0d8db236a287d657caca26073f6165d8"),
                    decoder,
                )
                .unwrap(),
            ],
        );
        let mut recognizer = Recognizer::new(
            Embedder::from_library("../../assets/nnfp_v3.weights").unwrap(),
            MusicDetector::from_library("../../assets/nnfp_v3.weights").unwrap(),
            shards,
            RecognitionPolicy::default(),
        )
        .unwrap();
        let wav = std::fs::read(wav_path).unwrap();
        let samples = wav[44..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| i16::from_le_bytes(*sample))
            .collect::<Vec<_>>();
        let outcome = recognizer.recognize_pcm_timed(&samples, 16_000).unwrap();
        eprintln!("{:?}", outcome.timings);
        let recognition = outcome.recognition.unwrap();
        assert_eq!(recognition.result.metadata.title, "Queen Tings");
        assert_eq!(recognition.result.metadata.artist, "Masego");
        assert!((recognition.result.score - 0.398_528_1).abs() <= 0.000_022_888);
        assert_eq!(recognition.result.offset_seconds, 36.0);
    }

    #[test]
    #[ignore = "requires NOWPLAYING_ARTIFACTS_PATH and NOWPLAYING_PREVIOUS_WAV"]
    fn reports_previous_match_diagnostics() {
        use std::sync::Arc;

        use crate::index::{MatcherConfig, Shard, ShardSet, TreeIdDecoder};

        let artifacts = std::env::var("NOWPLAYING_ARTIFACTS_PATH").unwrap();
        let samples = read_wav(std::env::var("NOWPLAYING_PREVIOUS_WAV").unwrap()).unwrap();
        let decoder =
            Arc::new(TreeIdDecoder::from_library("../../assets/nnfp_v3.weights").unwrap());
        let shards = ShardSet::new(
            MatcherConfig::from_file(format!("{artifacts}/etc/ambient/v3_config_tah.pb")).unwrap(),
            vec![
                Shard::open(
                    "matcher_tah.leveldb",
                    format!("{artifacts}/etc/ambient/matcher_tah.leveldb"),
                    decoder.clone(),
                )
                .unwrap(),
                Shard::open(
                    "US0d8db236a287d657caca26073f6165d8",
                    format!("{artifacts}/etc/ambient/US0d8db236a287d657caca26073f6165d8"),
                    decoder,
                )
                .unwrap(),
            ],
        );
        let mut recognizer = Recognizer::new(
            Embedder::from_library("../../assets/nnfp_v3.weights").unwrap(),
            MusicDetector::from_library("../../assets/nnfp_v3.weights").unwrap(),
            shards,
            RecognitionPolicy::default(),
        )
        .unwrap();
        for (sample_offset, predicted_offset, expected_offset, expected_score) in [
            (0, 36.0, 36.0, 0.810_297_97),
            (44_100, 38.0, 36.0, 0.740_305_8),
            (176_400, 44.0, 42.0, 0.775_112_5),
        ] {
            let previous = PreviousMatch {
                shard_name: "US0d8db236a287d657caca26073f6165d8".into(),
                track_id: 1871,
                predicted_offsets_seconds: vec![predicted_offset],
            };
            let outcome = recognizer
                .recognize_pcm_timed_with_context(
                    &samples[sample_offset..sample_offset + 176_400],
                    22_050,
                    true,
                    Some(&previous),
                )
                .unwrap();
            let continuity = outcome.continuity.unwrap();
            assert_eq!(continuity.track_id, 1871);
            assert_eq!(continuity.offset_seconds, expected_offset);
            assert!((continuity.score - expected_score).abs() <= 0.000_001);
        }
    }

    #[test]
    #[ignore = "requires NOWPLAYING_LIVE_PARITY_CATALOG"]
    fn matches_the_live_dsp_capture_oracle() {
        use std::path::PathBuf;
        use std::sync::Arc;

        use crate::index::{MatcherConfig, Shard, ShardSet, TreeIdDecoder};

        let catalog = PathBuf::from(std::env::var("NOWPLAYING_LIVE_PARITY_CATALOG").unwrap());
        let manifest_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/live-parity");
        let weights =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/nnfp_v3.weights");
        let decoder = Arc::new(TreeIdDecoder::from_library(&weights).unwrap());
        let mut shard_paths = std::fs::read_dir(&catalog)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("US"))
            })
            .collect::<Vec<_>>();
        shard_paths.sort_unstable();
        let mut shards = vec![
            Shard::open(
                "matcher_tah.leveldb",
                catalog.join("matcher_tah.leveldb"),
                decoder.clone(),
            )
            .unwrap(),
        ];
        for path in shard_paths {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            shards.push(Shard::open(name, path, decoder.clone()).unwrap());
        }
        let shards = ShardSet::new(
            MatcherConfig::from_file(catalog.join("v3_config_tah.pb")).unwrap(),
            shards,
        );
        let mut recognizer = Recognizer::new(
            Embedder::from_library(&weights).unwrap(),
            MusicDetector::from_library(&weights).unwrap(),
            shards,
            RecognitionPolicy::default(),
        )
        .unwrap();
        let manifest = std::fs::read_to_string(manifest_dir.join("manifest.tsv")).unwrap();
        for line in manifest.lines().skip(1) {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(fields.len(), 8);
            let capture = fields[0];
            let wav = std::fs::read(manifest_dir.join(capture)).unwrap();
            assert_eq!(&wav[..4], b"RIFF");
            assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
            let samples = wav[44..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| i16::from_le_bytes(*sample))
                .collect::<Vec<_>>();
            let recognition = recognizer.recognize_pcm(&samples, 16_000).unwrap();
            let expected_accepted = fields[1] == "1";
            assert_eq!(recognition.is_some(), expected_accepted, "{capture}");
            if !expected_accepted {
                continue;
            }
            let result = recognition.unwrap().result;
            assert_eq!(result.metadata.track_id, fields[2], "{capture}");
            assert_eq!(
                result.track_id,
                fields[3].parse::<u32>().unwrap(),
                "{capture}"
            );
            assert_eq!(result.shard_name, fields[4], "{capture}");
            assert_eq!(
                result.offset_seconds,
                fields[5].parse::<f64>().unwrap(),
                "{capture}"
            );
            let expected_score = fields[6].parse::<f32>().unwrap();
            assert!(
                (result.score - expected_score).abs() <= 0.000_03,
                "{capture}"
            );
        }
    }
}
