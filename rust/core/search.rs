//! IVFADC lookup and time-aligned search over matcher shards.
//!
//! Each 96-float embedding selects the nearest flat partitions. Search then builds a `12 * 256`
//! asymmetric squared-distance table and sums 12 lookups for every candidate code in those
//! partitions. Partition centroids only choose leaves. They are never subtracted from the query
//! before product-quantization distance is computed.
//!
//! Single-embedding results use distance, where smaller is better. Sequence results use the
//! recovered aggregate score, where larger is better. Acceptance requires a score strictly
//! greater than the matcher threshold and is left to [`crate::recognize::Recognizer`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::index::{
    CODEWORD_COUNT, EMBEDDING_SIZE, MatcherConfig, SUBQUANTIZER_COUNT, SUBSPACE_SIZE, ShardSet,
    TrackMetadata,
};
use crate::scorer::{SequenceScorer, similarity_metric};
use crate::{Error, Result};

/// A metadata-bearing match for one embedding, ordered by increasing distance.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Position of the owning shard in the searched [`ShardSet`].
    pub shard_index: usize,
    /// Stable label assigned when the owning shard was opened.
    pub shard_name: String,
    /// Shard-local numeric key used for metadata and reverse-index records.
    pub track_id: u32,
    /// Lowest asymmetric squared distance found for this track in the probed leaves.
    pub distance: f32,
    /// Display metadata decoded from the track's `M` family record.
    pub metadata: TrackMetadata,
}

/// A one-embedding match before its metadata record is loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchCandidate {
    /// Position of the owning shard in the searched [`ShardSet`].
    pub shard_index: usize,
    /// Shard-local numeric track ID.
    pub track_id: u32,
    /// Lowest asymmetric squared distance found for the track.
    pub distance: f32,
}

/// Nearest product-quantization representation of one query embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductQuantization {
    /// One selected codeword index for each contiguous eight-dimensional subspace.
    pub code: [u8; SUBQUANTIZER_COUNT],
    /// Summed squared L2 distance from the query to the reconstructed code.
    pub distance: f32,
}

/// The best time alignment scored for one candidate track.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceSearchResult {
    /// Position of the owning shard in the searched [`ShardSet`].
    pub shard_index: usize,
    /// Stable label assigned when the owning shard was opened.
    pub shard_name: String,
    /// Shard-local numeric track ID.
    pub track_id: u32,
    /// Aggregate sequence score, where larger values are better matches.
    pub score: f32,
    /// Stored-track offset selected by alignment, in one-second embedding steps.
    pub offset_seconds: f64,
    /// Display metadata decoded from the track's `M` family record.
    pub metadata: TrackMetadata,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct PreviousMatch {
    pub shard_name: String,
    pub track_id: u32,
    pub predicted_offsets_seconds: Vec<f64>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuityScore {
    pub shard_name: String,
    pub track_id: u32,
    pub score: f32,
    pub offset_seconds: f64,
}

/// Ranked sequence results and the intermediate values used to score them.
///
/// Results are ordered by descending score and can include rejected values at or below the
/// matcher threshold. The remaining vectors expose native-parity diagnostics for each query
/// embedding.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceSearch {
    /// Best alignment for each retained track, ordered by descending score.
    pub results: Vec<SequenceSearchResult>,
    /// Distinctiveness weight assigned to each query embedding.
    pub weights: Vec<f32>,
    /// Adaptive similarity metric assigned to each query embedding.
    pub similarity_metrics: Vec<f32>,
    /// Similarity-buffer lengths before the copied metric buffer is pruned to 201 values.
    pub neighbor_counts: Vec<usize>,
    /// Shared additive term used by every candidate sequence score.
    pub bias: f32,
    pub continuity: Option<ContinuityScore>,
}

#[derive(Debug, Clone, Copy)]
struct SequenceHit {
    shard_index: usize,
    track_id: u32,
    query_index: usize,
    partition: u16,
    partition_ordinal: usize,
    distance: f32,
}

const RAW_CANDIDATE_LIMIT: usize = 1_001;
const CANDIDATE_TRACK_LIMIT: usize = 40;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SequenceSearchTimings {
    pub search: Duration,
    pub score: Duration,
}

fn sort_three_by<T>(
    values: &mut [T],
    left: usize,
    middle: usize,
    right: usize,
    less: impl Fn(&T, &T) -> bool,
) -> usize {
    if !less(&values[middle], &values[left]) {
        if !less(&values[right], &values[middle]) {
            return 0;
        }
        values.swap(middle, right);
        if less(&values[middle], &values[left]) {
            values.swap(left, middle);
            return 2;
        }
        return 1;
    }
    if less(&values[right], &values[middle]) {
        values.swap(left, right);
        return 1;
    }
    values.swap(left, middle);
    if less(&values[right], &values[middle]) {
        values.swap(middle, right);
        return 2;
    }
    1
}

fn selection_sort_by<T>(
    values: &mut [T],
    first: usize,
    last: usize,
    less: impl Fn(&T, &T) -> bool,
) {
    for position in first..last.saturating_sub(1) {
        let mut minimum = position;
        for candidate in position + 1..last {
            if less(&values[candidate], &values[minimum]) {
                minimum = candidate;
            }
        }
        if minimum != position {
            values.swap(position, minimum);
        }
    }
}

fn libcxx_nth_element_by<T>(values: &mut [T], nth: usize, less: impl Fn(&T, &T) -> bool + Copy) {
    let mut first = 0;
    let mut last = values.len();
    loop {
        if nth == last {
            return;
        }
        let length = last - first;
        match length {
            0 | 1 => return,
            2 => {
                if less(&values[last - 1], &values[first]) {
                    values.swap(first, last - 1);
                }
                return;
            }
            3 => {
                sort_three_by(values, first, first + 1, last - 1, less);
                return;
            }
            4..=7 => {
                selection_sort_by(values, first, last, less);
                return;
            }
            _ => {}
        }

        let mut middle = first + length / 2;
        let last_minus_one = last - 1;
        let mut swaps = sort_three_by(values, first, middle, last_minus_one, less);
        let mut left = first;
        let mut right = last_minus_one;
        if !less(&values[left], &values[middle]) {
            let guarded = loop {
                right -= 1;
                if left == right {
                    break false;
                }
                if less(&values[right], &values[middle]) {
                    break true;
                }
            };
            if guarded {
                values.swap(left, right);
                swaps += 1;
            } else {
                left += 1;
                right = last - 1;
                if !less(&values[first], &values[right]) {
                    loop {
                        if left == right {
                            return;
                        }
                        if less(&values[first], &values[left]) {
                            values.swap(left, right);
                            swaps += 1;
                            left += 1;
                            break;
                        }
                        left += 1;
                    }
                }
                if left == right {
                    return;
                }
                loop {
                    while !less(&values[first], &values[left]) {
                        left += 1;
                    }
                    loop {
                        right -= 1;
                        if !less(&values[first], &values[right]) {
                            break;
                        }
                    }
                    if left >= right {
                        break;
                    }
                    values.swap(left, right);
                    swaps += 1;
                    left += 1;
                }
                if nth < left {
                    return;
                }
                first = left;
                continue;
            }
        }

        left += 1;
        if left < right {
            loop {
                while less(&values[left], &values[middle]) {
                    left += 1;
                }
                loop {
                    right -= 1;
                    if less(&values[right], &values[middle]) {
                        break;
                    }
                }
                if left >= right {
                    break;
                }
                values.swap(left, right);
                swaps += 1;
                if middle == left {
                    middle = right;
                }
                left += 1;
            }
        }
        if left != middle && less(&values[middle], &values[left]) {
            values.swap(left, middle);
            swaps += 1;
        }
        if nth == left {
            return;
        }
        if swaps == 0 {
            let (start, end) = if nth < left {
                (first, left)
            } else {
                (left, last)
            };
            if values[start..end]
                .windows(2)
                .all(|pair| !less(&pair[1], &pair[0]))
            {
                return;
            }
        }
        if nth < left {
            last = left;
        } else {
            first = left + 1;
        }
    }
}

fn prune_by<T>(values: &mut Vec<T>, limit: usize, less: impl Fn(&T, &T) -> bool + Copy) {
    libcxx_nth_element_by(values, limit - 1, less);
    values.truncate(limit);
}

/// Reconstructs the 96-float center selected by a 12-byte product code.
///
/// The result concatenates one eight-float codeword per subspace. No partition centroid is added
/// because the database codes and query lookup operate in the raw embedding space.
pub fn reconstruct_code(
    config: &MatcherConfig,
    code: &[u8; SUBQUANTIZER_COUNT],
) -> [f32; EMBEDDING_SIZE] {
    std::array::from_fn(|dimension| {
        let subspace = dimension / SUBSPACE_SIZE;
        let codeword = usize::from(code[subspace]);
        let codeword_start = (subspace * CODEWORD_COUNT + codeword) * SUBSPACE_SIZE;
        config.codebook[codeword_start + dimension % SUBSPACE_SIZE]
    })
}

/// Selects the nearest codeword in each subspace for a query embedding.
///
/// The returned distance is the sum of 12 squared L2 lookup values. This helper reproduces the
/// code format stored in leaves but is not needed for ordinary database queries.
pub fn quantize_query(config: &MatcherConfig, query: &[f32]) -> Result<ProductQuantization> {
    let lookup = asymmetric_lookup(config, query)?;
    let mut code = [0u8; SUBQUANTIZER_COUNT];
    for (subspace, selected) in code.iter_mut().enumerate() {
        let mut best = (0u8, f32::INFINITY);
        for codeword in 0..CODEWORD_COUNT {
            let distance = lookup[subspace * CODEWORD_COUNT + codeword];
            if distance < best.1 {
                best = (codeword as u8, distance);
            }
        }
        *selected = best.0;
    }
    let distance = lookup_distance(&lookup, &code);
    Ok(ProductQuantization { code, distance })
}

/// Builds the asymmetric squared-distance table for one query embedding.
///
/// The returned vector has `12 * 256` values in subspace-major order. Database search can score
/// any stored 12-byte code with one lookup per subspace.
pub fn asymmetric_lookup(config: &MatcherConfig, query: &[f32]) -> Result<Vec<f32>> {
    if query.len() != EMBEDDING_SIZE {
        return Err(Error::InvalidInput(format!(
            "search needs {EMBEDDING_SIZE} floats and received {}",
            query.len()
        )));
    }
    let mut lookup = Vec::with_capacity(SUBQUANTIZER_COUNT * CODEWORD_COUNT);
    for subspace in 0..SUBQUANTIZER_COUNT {
        let query_start = subspace * SUBSPACE_SIZE;
        let query = &query[query_start..query_start + SUBSPACE_SIZE];
        let query_norm = query_norm(query);
        for codeword in 0..CODEWORD_COUNT {
            let codeword_start = (subspace * CODEWORD_COUNT + codeword) * SUBSPACE_SIZE;
            let center = &config.codebook[codeword_start..codeword_start + SUBSPACE_SIZE];
            let mut dot = 0.0f32;
            for dimension in 0..SUBSPACE_SIZE {
                dot = query[dimension].mul_add(center[dimension], dot);
            }
            lookup.push((query_norm + center_norm(center)) + -2.0 * dot);
        }
    }
    Ok(lookup)
}

fn query_norm(values: &[f32]) -> f32 {
    let lanes = [
        values[0] * values[0] + values[4] * values[4],
        values[1] * values[1] + values[5] * values[5],
        values[2] * values[2] + values[6] * values[6],
        values[3] * values[3] + values[7] * values[7],
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

fn center_norm(values: &[f32]) -> f32 {
    let middle = (values[1] * values[1] + values[2] * values[2])
        + (values[3] * values[3] + values[4] * values[4]);
    let mut norm = values[0] * values[0] + middle;
    norm += values[5] * values[5];
    norm += values[6] * values[6];
    norm + values[7] * values[7]
}

fn lookup_distance(lookup: &[f32], code: &[u8; SUBQUANTIZER_COUNT]) -> f32 {
    code.iter()
        .copied()
        .enumerate()
        .fold(0.0f32, |distance, (subspace, codeword)| {
            distance + lookup[subspace * CODEWORD_COUNT + usize::from(codeword)]
        })
}

/// Selects the configured number of nearest flat partition centroids.
///
/// Results contain partition IDs and squared L2 distances, ordered from nearest to furthest with
/// the partition ID breaking exact ties.
pub fn nearest_partitions(config: &MatcherConfig, query: &[f32]) -> Result<Vec<(u32, f32)>> {
    if query.len() != EMBEDDING_SIZE {
        return Err(Error::InvalidInput(format!(
            "search needs {EMBEDDING_SIZE} floats and received {}",
            query.len()
        )));
    }
    let mut distances = (0..config.partition_count)
        .map(|partition| {
            let centroid =
                &config.centroids[partition * EMBEDDING_SIZE..(partition + 1) * EMBEDDING_SIZE];
            let distance = query
                .iter()
                .zip(centroid)
                .fold(0.0f32, |sum, (&left, &right)| {
                    let difference = left - right;
                    difference.mul_add(difference, sum)
                });
            (partition as u32, distance)
        })
        .collect::<Vec<_>>();
    distances.sort_unstable_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    distances.truncate(config.probe_count);
    Ok(distances)
}

/// Scores one stored product code against a query embedding.
///
/// `partition` is validated because it identifies the leaf that owns the code. Its centroid is
/// deliberately not subtracted from the query, matching Google's asymmetric search path.
pub fn asymmetric_distance(
    config: &MatcherConfig,
    query: &[f32],
    partition: u32,
    code: &[u8; SUBQUANTIZER_COUNT],
) -> Result<f32> {
    if query.len() != EMBEDDING_SIZE {
        return Err(Error::InvalidInput(format!(
            "search needs {EMBEDDING_SIZE} floats and received {}",
            query.len()
        )));
    }
    let partition = usize::try_from(partition)
        .map_err(|_| Error::InvalidInput("partition does not fit usize".into()))?;
    if partition >= config.partition_count {
        return Err(Error::InvalidInput(format!(
            "partition {partition} is out of range"
        )));
    }
    Ok(lookup_distance(&asymmetric_lookup(config, query)?, code))
}

/// Searches every installed shard for the nearest tracks to one embedding.
///
/// Each `(shard, track)` pair keeps its lowest distance across all probed occurrences. At most
/// `limit` metadata-bearing results are returned, ordered by increasing distance.
pub fn search(shards: &ShardSet, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
    let mut ranked = search_candidates(shards, query)?;
    ranked.truncate(limit);
    ranked
        .into_iter()
        .filter_map(|candidate| {
            let shard = &shards[candidate.shard_index];
            match shard.track(candidate.track_id) {
                Ok(Some(metadata)) => Some(Ok(SearchResult {
                    shard_index: candidate.shard_index,
                    shard_name: shard.name.clone(),
                    track_id: candidate.track_id,
                    distance: candidate.distance,
                    metadata,
                })),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

/// Returns every unique one-embedding candidate without loading metadata.
///
/// Candidates from all shards are ordered by increasing distance. Equal distances are ordered by
/// shard position and numeric track ID.
pub fn search_candidates(shards: &ShardSet, query: &[f32]) -> Result<Vec<SearchCandidate>> {
    let partitions = nearest_partitions(&shards.config, query)?;
    let lookup = asymmetric_lookup(&shards.config, query)?;
    let mut candidates = HashMap::<(usize, u32), f32>::new();
    for (shard_index, shard) in shards.iter().enumerate() {
        for &(partition, _) in &partitions {
            let Some(leaf) = shard.leaf_shared(partition)? else {
                continue;
            };
            for (&track_id, code) in leaf.track_ids.iter().zip(&leaf.codes) {
                let distance = lookup_distance(&lookup, code);
                candidates
                    .entry((shard_index, track_id))
                    .and_modify(|current| *current = current.min(distance))
                    .or_insert(distance);
            }
        }
    }

    let mut ranked = candidates
        .into_iter()
        .map(|((shard_index, track_id), distance)| SearchCandidate {
            shard_index,
            track_id,
            distance,
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        left.distance.total_cmp(&right.distance).then_with(|| {
            (left.shard_index, left.track_id).cmp(&(right.shard_index, right.track_id))
        })
    });
    Ok(ranked)
}

/// Aligns an embedding sequence against stored track occurrences and ranks aggregate scores.
///
/// Every query probes every shard. Candidate and similarity buffers reproduce the native libc++
/// partition order before a first-pass score retains 40 tracks. Only offsets represented by raw
/// hits are scored in the final pass, and each offset step represents one second of source audio.
///
/// Scores are returned even when they fail the strict acceptance threshold. Empty embeddings or
/// a zero limit return an empty result with negative infinite bias.
pub fn search_sequences(
    shards: &ShardSet,
    embeddings: &[[f32; EMBEDDING_SIZE]],
    limit: usize,
) -> Result<SequenceSearch> {
    search_sequences_timed(shards, embeddings, limit).map(|(search, _)| search)
}

pub(crate) fn search_sequences_timed(
    shards: &ShardSet,
    embeddings: &[[f32; EMBEDDING_SIZE]],
    limit: usize,
) -> Result<(SequenceSearch, SequenceSearchTimings)> {
    search_sequences_timed_with_previous(shards, embeddings, limit, None)
}

pub(crate) fn search_sequences_timed_with_previous(
    shards: &ShardSet,
    embeddings: &[[f32; EMBEDDING_SIZE]],
    limit: usize,
    previous_match: Option<&PreviousMatch>,
) -> Result<(SequenceSearch, SequenceSearchTimings)> {
    if embeddings.is_empty() || limit == 0 {
        return Ok((
            SequenceSearch {
                results: Vec::new(),
                weights: Vec::new(),
                similarity_metrics: Vec::new(),
                neighbor_counts: Vec::new(),
                bias: f32::NEG_INFINITY,
                continuity: None,
            },
            SequenceSearchTimings::default(),
        ));
    }
    let search_started = Instant::now();
    let mut hits_by_track = HashMap::<(usize, u32), Vec<SequenceHit>>::new();
    let lookups = embeddings
        .iter()
        .map(|query| asymmetric_lookup(&shards.config, query))
        .collect::<Result<Vec<_>>>()?;
    let mut similarity_metrics = Vec::with_capacity(embeddings.len());
    let mut neighbor_counts = Vec::with_capacity(embeddings.len());
    let worker_count = embeddings.len().clamp(1, 4);
    let mut query_results = std::thread::scope(|scope| {
        let lookups = &lookups;
        let mut workers = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            workers.push(scope.spawn(move || {
                embeddings
                    .iter()
                    .zip(lookups.iter())
                    .enumerate()
                    .skip(worker_index)
                    .step_by(worker_count)
                    .map(|(index, (query, lookup))| {
                        search_sequence_query(shards, index, query, lookup)
                            .map(|result| (index, result))
                    })
                    .collect::<Result<Vec<_>>>()
            }));
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| Error::Format("sequence query worker panicked".into()))?
            })
            .collect::<Result<Vec<Vec<_>>>>()
    })?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    query_results.sort_unstable_by_key(|(index, _)| *index);
    for (_, (raw_candidates, similarity_metric, neighbor_count)) in query_results {
        similarity_metrics.push(similarity_metric);
        neighbor_counts.push(neighbor_count);
        for hit in raw_candidates {
            hits_by_track
                .entry((hit.shard_index, hit.track_id))
                .or_default()
                .push(hit);
        }
    }

    let search = search_started.elapsed();
    let score_started = Instant::now();
    let scorer = SequenceScorer::new(
        shards.config.scorer,
        embeddings,
        similarity_metrics,
        shards.background_mass(),
    )?;
    let continuity = previous_match
        .map(|previous| score_previous_match(shards, &lookups, previous))
        .transpose()?
        .flatten();
    let mut shortlisted = hits_by_track
        .into_iter()
        .map(|((shard_index, track_id), hits)| {
            let mut distances = vec![f32::INFINITY; embeddings.len()];
            for hit in &hits {
                distances[hit.query_index] = distances[hit.query_index].min(hit.distance);
            }
            scorer
                .score(&distances)
                .map(|score| (shard_index, track_id, score, hits))
        })
        .collect::<Result<Vec<_>>>()?;
    shortlisted.sort_unstable_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| (left.0, left.1).cmp(&(right.0, right.1)))
    });
    shortlisted.truncate(CANDIDATE_TRACK_LIMIT);
    let mut shortlisted_by_shard = shards.iter().map(|_| Vec::new()).collect::<Vec<_>>();
    for (shard_index, track_id, _, hits) in shortlisted {
        shortlisted_by_shard[shard_index].push((track_id, hits));
    }
    let worker_count = shortlisted_by_shard.len().clamp(1, 4);
    let next_shard = AtomicUsize::new(0);
    let mut scored = std::thread::scope(|scope| {
        let lookups = &lookups;
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let shortlisted_by_shard = &shortlisted_by_shard;
            let scorer = &scorer;
            let next_shard = &next_shard;
            workers.push(scope.spawn(move || {
                let mut scored = Vec::new();
                loop {
                    let shard_index = next_shard.fetch_add(1, Ordering::Relaxed);
                    if shard_index >= shortlisted_by_shard.len() {
                        break;
                    }
                    let shard = &shards[shard_index];
                    let track_hits = &shortlisted_by_shard[shard_index];
                    let track_ids = track_hits
                        .iter()
                        .map(|(track_id, _)| *track_id)
                        .collect::<Vec<_>>();
                    let partitions_by_track = shard.track_partitions_batch(&track_ids)?;
                    for (track_id, hits) in track_hits {
                        let Some(partitions) = partitions_by_track.get(track_id) else {
                            continue;
                        };
                        let mut occurrences = HashMap::<u16, Vec<usize>>::new();
                        let mut occurrence_counts = HashMap::<u16, usize>::new();
                        let mut occurrence_ordinals = Vec::with_capacity(partitions.len());
                        for (occurrence, &partition) in partitions.iter().enumerate() {
                            occurrences.entry(partition).or_default().push(occurrence);
                            let ordinal = occurrence_counts.entry(partition).or_default();
                            occurrence_ordinals.push(*ordinal);
                            *ordinal += 1;
                        }
                        let mut alignments = HashMap::<i64, Vec<f32>>::new();
                        for hit in hits {
                            let Some(&occurrence) = occurrences
                                .get(&hit.partition)
                                .and_then(|values| values.get(hit.partition_ordinal))
                            else {
                                continue;
                            };
                            let offset = occurrence as i64 - hit.query_index as i64;
                            let distances = alignments
                                .entry(offset)
                                .or_insert_with(|| vec![f32::INFINITY; embeddings.len()]);
                            distances[hit.query_index] =
                                distances[hit.query_index].min(hit.distance);
                        }
                        let mut leaves = HashMap::new();
                        for (offset, distances) in &mut alignments {
                            for (query_index, distance) in distances.iter_mut().enumerate() {
                                let Ok(occurrence) = usize::try_from(*offset + query_index as i64)
                                else {
                                    *distance = f32::INFINITY;
                                    continue;
                                };
                                let Some((&partition, &ordinal)) = partitions
                                    .get(occurrence)
                                    .zip(occurrence_ordinals.get(occurrence))
                                else {
                                    *distance = f32::INFINITY;
                                    continue;
                                };
                                let leaf = match leaves.entry(partition) {
                                    std::collections::hash_map::Entry::Occupied(entry) => {
                                        entry.into_mut()
                                    }
                                    std::collections::hash_map::Entry::Vacant(entry) => {
                                        let leaf = shard
                                            .leaf_shared(u32::from(partition))?
                                            .ok_or_else(|| {
                                                Error::Format(format!(
                                                    "reverse index references missing partition {partition}"
                                                ))
                                            })?;
                                        entry.insert(leaf)
                                    }
                                };
                                let first = leaf
                                    .track_ids
                                    .partition_point(|&candidate| candidate < *track_id);
                                let code = leaf
                                    .track_ids
                                    .get(first + ordinal)
                                    .filter(|&&candidate| candidate == *track_id)
                                    .and_then(|_| leaf.codes.get(first + ordinal))
                                    .ok_or_else(|| {
                                        Error::Format(format!(
                                            "reverse index occurrence {occurrence} for track {track_id} has no tree code"
                                        ))
                                    })?;
                                *distance = lookup_distance(&lookups[query_index], code);
                            }
                        }
                        let mut best = None;
                        for (offset, distances) in alignments {
                            let score = scorer.score(&distances)?;
                            if best.is_none_or(|(_, current)| score > current) {
                                best = Some((offset, score));
                            }
                        }
                        let Some((offset, score)) = best else {
                            continue;
                        };
                        scored.push((shard_index, *track_id, score, offset));
                    }
                }
                Ok(scored)
            }));
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| Error::Format("sequence scoring worker panicked".into()))?
            })
            .collect::<Result<Vec<Vec<_>>>>()
    })?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    scored.sort_unstable_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| (left.0, left.1).cmp(&(right.0, right.1)))
    });
    scored.truncate(limit);
    let mut results = Vec::with_capacity(scored.len());
    for (shard_index, track_id, score, offset) in scored {
        let shard = &shards[shard_index];
        let Some(metadata) = shard.track(track_id)? else {
            continue;
        };
        results.push(SequenceSearchResult {
            shard_index,
            shard_name: shard.name.clone(),
            track_id,
            score,
            offset_seconds: offset as f64,
            metadata,
        });
    }
    Ok((
        SequenceSearch {
            results,
            weights: scorer.weights().to_vec(),
            similarity_metrics: scorer.similarity_metrics().to_vec(),
            neighbor_counts,
            bias: scorer.bias(),
            continuity,
        },
        SequenceSearchTimings {
            search,
            score: score_started.elapsed(),
        },
    ))
}

fn score_previous_match(
    shards: &ShardSet,
    lookups: &[Vec<f32>],
    previous: &PreviousMatch,
) -> Result<Option<ContinuityScore>> {
    let Some((shard_index, shard)) = shards
        .iter()
        .enumerate()
        .find(|(_, shard)| shard.name == previous.shard_name)
    else {
        return Ok(None);
    };
    let Some(lookup) = lookups.first() else {
        return Ok(None);
    };
    let Some(codes) = shard.track_codes_shared(previous.track_id)? else {
        return Ok(None);
    };
    let mut candidate_offsets = HashMap::new();
    for predicted in &previous.predicted_offsets_seconds {
        if !predicted.is_finite() {
            continue;
        }
        let predicted = predicted.round() as i64;
        for offset in predicted - 2..=predicted + 2 {
            candidate_offsets.insert(offset, ());
        }
    }
    let mut best = None;
    for offset in candidate_offsets.into_keys() {
        let Ok(occurrence) = usize::try_from(offset) else {
            continue;
        };
        let Some(stored) = codes.get(occurrence) else {
            continue;
        };
        let score = 2.0 - lookup_distance(lookup, &stored.code);
        if best.is_none_or(|(_, current)| score > current) {
            best = Some((offset, score));
        }
    }
    Ok(best.map(|(offset, score)| ContinuityScore {
        shard_name: shards[shard_index].name.clone(),
        track_id: previous.track_id,
        score,
        offset_seconds: offset as f64,
    }))
}

fn search_sequence_query(
    shards: &ShardSet,
    query_index: usize,
    query: &[f32; EMBEDDING_SIZE],
    lookup: &[f32],
) -> Result<(Vec<SequenceHit>, f32, usize)> {
    let mut partitions = nearest_partitions(&shards.config, query)?;
    partitions.sort_unstable_by_key(|&(partition, _)| partition);
    let mut hits = Vec::new();
    for (shard_index, shard) in shards.iter().enumerate() {
        for &(partition, _) in &partitions {
            let Some(leaf) = shard.leaf_shared(partition)? else {
                continue;
            };
            let mut previous_track = None;
            let mut partition_ordinal = 0;
            for (&track_id, code) in leaf.track_ids.iter().zip(&leaf.codes) {
                if previous_track == Some(track_id) {
                    partition_ordinal += 1;
                } else {
                    previous_track = Some(track_id);
                    partition_ordinal = 0;
                }
                hits.push(SequenceHit {
                    shard_index,
                    track_id,
                    query_index,
                    partition: partition as u16,
                    partition_ordinal,
                    distance: lookup_distance(lookup, code),
                });
            }
        }
    }
    let mut raw_candidates = Vec::with_capacity(RAW_CANDIDATE_LIMIT * 2);
    let mut raw_threshold = f32::INFINITY;
    for hit in hits {
        if hit.distance >= raw_threshold {
            continue;
        }
        raw_candidates.push(hit);
        if raw_candidates.len() >= RAW_CANDIDATE_LIMIT * 2 {
            prune_by(&mut raw_candidates, RAW_CANDIDATE_LIMIT, |left, right| {
                left.distance < right.distance
            });
            raw_threshold = raw_candidates[RAW_CANDIDATE_LIMIT - 1].distance;
        }
    }
    if raw_candidates.len() > RAW_CANDIDATE_LIMIT {
        prune_by(&mut raw_candidates, RAW_CANDIDATE_LIMIT, |left, right| {
            left.distance < right.distance
        });
    }
    let retained_limit = shards.config.scorer.nearest_neighbor_count + 1;
    let mut nearest_distances = Vec::with_capacity(retained_limit * 2);
    let mut threshold = f32::INFINITY;
    for hit in &raw_candidates {
        if hit.distance >= threshold {
            continue;
        }
        nearest_distances.push(hit.distance);
        if nearest_distances.len() >= retained_limit * 2 {
            prune_by(&mut nearest_distances, retained_limit, |left, right| {
                left < right
            });
            threshold = nearest_distances[retained_limit - 1];
        }
    }
    let neighbor_count = nearest_distances.len();
    if nearest_distances.len() > retained_limit {
        prune_by(&mut nearest_distances, retained_limit, |left, right| {
            left < right
        });
    }
    let minimum_distance = (0..SUBQUANTIZER_COUNT)
        .map(|subspace| {
            lookup[subspace * CODEWORD_COUNT..(subspace + 1) * CODEWORD_COUNT]
                .iter()
                .copied()
                .min_by(f32::total_cmp)
                .unwrap_or(f32::INFINITY)
        })
        .sum();
    let metric = similarity_metric(
        shards.config.scorer,
        minimum_distance,
        &nearest_distances,
        shards.background_mass(),
    );
    Ok((raw_candidates, metric, neighbor_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MatcherConfig {
        let mut codebook = vec![0.0; SUBQUANTIZER_COUNT * CODEWORD_COUNT * SUBSPACE_SIZE];
        for subspace in 0..SUBQUANTIZER_COUNT {
            for dimension in 0..SUBSPACE_SIZE {
                codebook[((subspace * CODEWORD_COUNT + 1) * SUBSPACE_SIZE) + dimension] = 1.0;
            }
        }
        MatcherConfig::new(2, 1, codebook, vec![0.0; 2 * EMBEDDING_SIZE]).unwrap()
    }

    #[test]
    fn partition_ranking_is_known_by_construction() {
        let mut config = config();
        config.centroids[EMBEDDING_SIZE..].fill(2.0);
        let query = vec![1.75; EMBEDDING_SIZE];
        assert_eq!(nearest_partitions(&config, &query).unwrap()[0].0, 1);
    }

    #[test]
    fn asymmetric_code_ranking_is_known_by_construction() {
        let mut config = config();
        config.centroids[..EMBEDDING_SIZE].fill(4.0);
        let query = (0..EMBEDDING_SIZE)
            .map(|dimension| dimension as f32 / EMBEDDING_SIZE as f32)
            .collect::<Vec<_>>();
        for subspace in 0..SUBQUANTIZER_COUNT {
            for dimension in 0..SUBSPACE_SIZE {
                let query_dimension = subspace * SUBSPACE_SIZE + dimension;
                config.codebook[((subspace * CODEWORD_COUNT + 1) * SUBSPACE_SIZE) + dimension] =
                    query[query_dimension];
            }
        }
        let zero = asymmetric_distance(&config, &query, 0, &[0; 12]).unwrap();
        let one = asymmetric_distance(&config, &query, 0, &[1; 12]).unwrap();
        assert_eq!(one.to_bits(), 0x3518_8000);
        assert!(zero > one);
    }

    #[test]
    fn libcxx_partition_retains_the_lowest_values() {
        let mut values = (0..2_002).rev().collect::<Vec<_>>();
        prune_by(&mut values, RAW_CANDIDATE_LIMIT, |left, right| left < right);
        values.sort_unstable();
        assert_eq!(values, (0..RAW_CANDIDATE_LIMIT).collect::<Vec<_>>());
    }

    #[test]
    fn reconstructed_code_has_zero_distance() {
        let config = config();
        let code = std::array::from_fn(|index| (index % 2) as u8);
        let query = reconstruct_code(&config, &code);
        assert_eq!(asymmetric_distance(&config, &query, 0, &code).unwrap(), 0.0);
    }

    #[test]
    fn product_quantization_selects_the_nearest_codewords() {
        let mut config = config();
        for subspace in 0..SUBQUANTIZER_COUNT {
            for dimension in 0..SUBSPACE_SIZE {
                config.codebook[((subspace * CODEWORD_COUNT + 2) * SUBSPACE_SIZE) + dimension] =
                    2.0;
            }
        }
        let query = vec![1.75; EMBEDDING_SIZE];
        let quantized = quantize_query(&config, &query).unwrap();
        assert_eq!(quantized.code, [2; SUBQUANTIZER_COUNT]);
        assert_eq!(quantized.distance, 6.0);
    }
}
