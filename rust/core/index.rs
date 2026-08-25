//! Matcher configuration and song-shard decoding.
//!
//! Google's matcher data combines one protobuf configuration with independent standalone
//! LevelDB table files. The configuration holds 1,000 partition centroids, a `12 * 256 * 8`
//! product-quantization codebook, and sequence-scorer constants. Every shard contains its own
//! metadata, reverse index, and candidate records in the same quantization space.
//!
//! Shards are additive rather than divisions of one table. Search probes every installed shard,
//! so coverage improves as shards are added while lookup cost grows linearly with catalog size.
//! The table bytes are downloaded from Google's public bucket and must be treated as untrusted.

use std::collections::HashMap;
use std::fs;
use std::ops::Index;
use std::path::Path;
use std::sync::Arc;

use crate::sstable::{CacheKey, DEFAULT_CACHE_BYTES, DecodedCache, Table};
use crate::{Error, Result};

/// Number of flat IVF partitions in the production matcher.
pub const PARTITION_COUNT: usize = 1_000;
/// Nearest partitions searched in each installed shard for one embedding.
pub const PROBE_COUNT: usize = 20;
/// Product-quantization subspaces covering one embedding.
pub const SUBQUANTIZER_COUNT: usize = 12;
/// Candidate reconstruction vectors available in each subspace.
pub const CODEWORD_COUNT: usize = 256;
/// Contiguous embedding dimensions covered by one subspace.
pub const SUBSPACE_SIZE: usize = 8;
/// Dimensions in a matcher embedding.
pub const EMBEDDING_SIZE: usize = SUBQUANTIZER_COUNT * SUBSPACE_SIZE;
const SQUARED_L2_DISTANCE: u64 = 0;

/// Quantizer and scorer parameters decoded from `v3_config_tah.pb`.
///
/// The codebook is laid out by subspace, codeword, then dimension. Centroids are laid out by
/// partition then dimension. Both arrays operate directly on the raw 96-float embedding.
#[derive(Debug, Clone)]
pub struct MatcherConfig {
    /// Number of 96-dimensional flat partition centroids.
    pub partition_count: usize,
    /// Number of nearest partitions searched per shard.
    pub probe_count: usize,
    /// Product-quantization vectors in `subspace * 256 * 8` order.
    pub codebook: Vec<f32>,
    /// Flat partition centroids in `partition * 96` order.
    pub centroids: Vec<f32>,
    /// Constants used to turn an aligned distance sequence into one score.
    pub scorer: ScorerConfig,
    /// Strict lower bound for accepting a sequence score.
    pub acceptance_threshold: f32,
}

/// Constants recovered from Google's sequence scorer configuration.
///
/// The fields parameterize distinctiveness weights, the adaptive neighbor metric, and the final
/// aggregate score. They are exposed so traced configurations can be replayed without changing
/// the scoring implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScorerConfig {
    /// Nearest distances below the extra retained boundary used by the adaptive metric.
    pub nearest_neighbor_count: usize,
    /// Multiplier applied to background mass before neighbor-count normalization.
    pub neighbor_mass_scale: f32,
    /// Multiplier applied to each aligned product-quantization distance.
    pub distance_scale: f32,
    /// Additive intercept in the adaptive similarity metric.
    pub similarity_bias: f32,
    /// Multiplier applied to pairwise embedding squared distance when deriving query weights.
    pub distinctiveness_scale: f32,
    /// Offset subtracted from scaled pairwise distance when deriving query weights.
    pub distinctiveness_bias: f32,
    /// Exponent denominator used to adapt the retained-neighbor threshold.
    pub neighbor_exponent: f32,
    /// Multiplier converting the adaptive distance threshold into a similarity metric.
    pub similarity_scale: f32,
    /// Input to the softplus term that normalizes aggregate query weight.
    pub weight_sum_shape: f32,
    /// Constant added when deriving the scorer bias from background mass.
    pub score_offset: f32,
}

/// Display metadata decoded from an `M` family shard record.
///
/// Optional protobuf fields become empty strings or `None`. The numeric key used inside a shard
/// is separate from [`Self::track_id`], which is the catalog's string identifier.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackMetadata {
    /// Stable string identifier used by the Java history and metadata paths.
    pub track_id: String,
    /// Opaque catalog identifier whose upstream namespace remains unknown.
    pub catalog_id: String,
    /// Default display title stored in protobuf field 3.
    pub title: String,
    /// Default display artist stored in protobuf field 4.
    pub artist: String,
    /// Track duration decoded from the optional fixed64 double field.
    pub duration_seconds: Option<f64>,
    /// Knowledge Graph `/m/` or `/g/` identifier used for identity.
    pub knowledge_graph_id: String,
    /// Album name when the shard carries richer country metadata.
    pub album: String,
    /// Release year when present in the shard metadata.
    pub release_year: Option<u32>,
}

/// Candidate occurrences decoded from one `T` family partition record.
///
/// The vectors are parallel. Each 12-byte code belongs to the track ID at the same position, and
/// repeated track IDs preserve distinct stored occurrences.
#[derive(Debug, Clone)]
pub struct SearchLeaf {
    /// Product-quantization codes in leaf occurrence order.
    pub codes: Vec<[u8; SUBQUANTIZER_COUNT]>,
    /// Delta-Huffman-decoded numeric track IDs in leaf occurrence order.
    pub track_ids: Vec<u32>,
}

/// One track occurrence resolved through its reverse index and partition leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredCode {
    /// Zero-based IVF partition containing the occurrence.
    pub partition: u32,
    /// Twelve codeword selectors covering all 96 embedding dimensions.
    pub code: [u8; SUBQUANTIZER_COUNT],
}

/// Decoder for the static Huffman stream used by `T` family track IDs.
///
/// Its tables are model data recovered from fixed offsets in the NNFP native layout. Decoded
/// symbols are nonnegative deltas whose running sum produces nondecreasing numeric track IDs.
#[derive(Debug, Clone)]
pub struct TreeIdDecoder {
    primary: [u32; 256],
    fast: [u8; 1_024],
    long_codes: [u32; 210],
    long_symbols: [u8; 210],
}

/// One self-contained matcher shard backed by a standalone LevelDB table file.
///
/// Records use `M` for metadata, `R` for track-to-partition occurrences, and `T` for partition
/// candidate lists. The four-byte key stores the family followed by a 24-bit little-endian ID.
#[derive(Debug)]
pub struct Shard {
    /// Stable label copied into search results, usually the shard filename.
    pub name: String,
    table: Table,
    decoder: Arc<TreeIdDecoder>,
}

/// Reusable collection of independently searchable catalog shards.
///
/// Borrowed iteration and indexed access leave the set reusable. Each query visits every shard,
/// each shard carries all 1,000 candidate partitions, and all shards must use the shared matcher
/// configuration. Search cost therefore grows linearly with catalog size.
#[derive(Debug)]
pub struct ShardSet {
    /// Quantizer and scorer configuration shared by every shard in the set.
    pub config: MatcherConfig,
    shards: Vec<Shard>,
    background_mass: f32,
}

#[derive(Clone, Copy)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64(&'a [u8]),
    Bytes(&'a [u8]),
    Fixed32(&'a [u8]),
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            nearest_neighbor_count: 200,
            neighbor_mass_scale: 8.863_962e-5,
            distance_scale: 19.368_963,
            similarity_bias: -0.575_498_64,
            distinctiveness_scale: 1.452_597_3,
            distinctiveness_bias: 0.634_999_3,
            neighbor_exponent: 16.925_064,
            similarity_scale: 25.503_857,
            weight_sum_shape: -1.726_040_4,
            score_offset: 0.777_521_2,
        }
    }
}

impl MatcherConfig {
    /// Decodes and validates the production matcher protobuf.
    ///
    /// Google's serialized probe field is twice the number of partitions searched by the native
    /// query path, so this parser converts the stored value 40 to [`PROBE_COUNT`]. Unsupported
    /// distance measures and array shapes return [`Error::Format`].
    pub fn parse(data: &[u8]) -> Result<Self> {
        let outer = fields(data)?;
        let matcher = fields(one_bytes(&outer, 13)?)?;
        let partition_count = usize::try_from(one_varint(&matcher, 1)?)
            .map_err(|_| Error::Format("partition count does not fit usize".into()))?;
        let serialized_probe_count = usize::try_from(one_varint(&matcher, 2)?)
            .map_err(|_| Error::Format("probe count does not fit usize".into()))?;
        if serialized_probe_count % 2 != 0 {
            return Err(Error::Format("serialized probe count is not even".into()));
        }
        let probe_count = serialized_probe_count / 2;
        let scorer_fields = fields(one_bytes(&matcher, 6)?)?;
        let scorer = ScorerConfig {
            nearest_neighbor_count: usize::try_from(one_varint(&scorer_fields, 7)?)
                .map_err(|_| Error::Format("nearest neighbor count does not fit usize".into()))?,
            neighbor_mass_scale: one_f32(&scorer_fields, 8)?,
            distance_scale: one_f32(&scorer_fields, 9)?,
            similarity_bias: one_f32(&scorer_fields, 10)?,
            distinctiveness_scale: one_f32(&scorer_fields, 11)?,
            distinctiveness_bias: one_f32(&scorer_fields, 12)?,
            neighbor_exponent: one_f32(&scorer_fields, 13)?,
            similarity_scale: one_f32(&scorer_fields, 14)?,
            weight_sum_shape: one_f32(&scorer_fields, 15)?,
            score_offset: one_f32(&scorer_fields, 16)?,
        };
        let acceptance_threshold = optional_f32(&outer, 19)?.unwrap_or(0.0);

        let asymmetric_hashing = fields(one_bytes(&matcher, 3)?)?;
        if one_varint(&asymmetric_hashing, 2)? != SQUARED_L2_DISTANCE {
            return Err(Error::Format(
                "unsupported asymmetric hashing distance measure".into(),
            ));
        }
        let subspaces = repeated_bytes(&asymmetric_hashing, 1);
        let mut codebook = Vec::new();
        for subspace in subspaces {
            let subspace = fields(subspace)?;
            let entries = repeated_bytes(&subspace, 1);
            if entries.len() != CODEWORD_COUNT {
                return Err(Error::Format(format!(
                    "codebook subspace has {} entries",
                    entries.len()
                )));
            }
            for entry in entries {
                let entry = fields(entry)?;
                codebook.extend(parse_packed_f32(one_bytes(&entry, 1)?, SUBSPACE_SIZE)?);
            }
        }

        let partitioner = fields(one_bytes(&matcher, 4)?)?;
        if one_varint(&partitioner, 3)? != SQUARED_L2_DISTANCE {
            return Err(Error::Format(
                "unsupported partition distance measure".into(),
            ));
        }
        let leaves = repeated_bytes(&partitioner, 1);
        let mut centroids = Vec::new();
        for leaf in leaves {
            let leaf = fields(leaf)?;
            centroids.extend(parse_packed_f32(one_bytes(&leaf, 1)?, EMBEDDING_SIZE)?);
        }

        Self::new_with_scorer(
            partition_count,
            probe_count,
            codebook,
            centroids,
            scorer,
            acceptance_threshold,
        )
    }

    /// Reads and parses a matcher protobuf from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&fs::read(path)?)
    }

    /// Builds a matcher configuration with the recovered production scorer constants.
    ///
    /// `codebook` must contain `12 * 256 * 8` floats and `centroids` must contain
    /// `partition_count * 96` floats. The acceptance threshold defaults to a strict zero.
    pub fn new(
        partition_count: usize,
        probe_count: usize,
        codebook: Vec<f32>,
        centroids: Vec<f32>,
    ) -> Result<Self> {
        Self::new_with_scorer(
            partition_count,
            probe_count,
            codebook,
            centroids,
            ScorerConfig::default(),
            0.0,
        )
    }

    /// Builds and validates a matcher configuration with explicit scoring parameters.
    ///
    /// The codebook and centroids use the layouts described by [`MatcherConfig`]. A zero
    /// partition count, invalid probe count, empty neighbor set, or non-finite threshold is
    /// rejected as malformed configuration.
    pub fn new_with_scorer(
        partition_count: usize,
        probe_count: usize,
        codebook: Vec<f32>,
        centroids: Vec<f32>,
        scorer: ScorerConfig,
        acceptance_threshold: f32,
    ) -> Result<Self> {
        if partition_count == 0 || probe_count == 0 || probe_count > partition_count {
            return Err(Error::Format("matcher partition counts are invalid".into()));
        }
        if codebook.len() != SUBQUANTIZER_COUNT * CODEWORD_COUNT * SUBSPACE_SIZE {
            return Err(Error::Format(format!(
                "codebook has {} floats",
                codebook.len()
            )));
        }
        if centroids.len() != partition_count * EMBEDDING_SIZE {
            return Err(Error::Format(format!(
                "partitioner has {} floats",
                centroids.len()
            )));
        }
        if scorer.nearest_neighbor_count == 0 || !acceptance_threshold.is_finite() {
            return Err(Error::Format(
                "matcher scorer configuration is invalid".into(),
            ));
        }
        Ok(Self {
            partition_count,
            probe_count,
            codebook,
            centroids,
            scorer,
            acceptance_threshold,
        })
    }
}

impl ShardSet {
    /// Associates a shared matcher configuration with independently searchable shards.
    pub fn new(config: MatcherConfig, mut shards: Vec<Shard>) -> Self {
        let background_mass = shards.iter().map(Shard::background_mass).sum();
        let cache = Arc::new(DecodedCache::new(Some(DEFAULT_CACHE_BYTES)));
        for shard in &mut shards {
            shard.table.install_cache(Arc::clone(&cache));
        }
        Self {
            config,
            shards,
            background_mass,
        }
    }

    /// Iterates over shards in lookup order without consuming the collection.
    pub fn iter(&self) -> std::slice::Iter<'_, Shard> {
        self.shards.iter()
    }

    pub(crate) fn discard_mapped_pages(&self) {
        for shard in &self.shards {
            shard.table.discard_mapped_pages();
        }
    }

    pub(crate) fn background_mass(&self) -> f32 {
        self.background_mass
    }
}

impl Index<usize> for ShardSet {
    type Output = Shard;

    fn index(&self, index: usize) -> &Self::Output {
        &self.shards[index]
    }
}

impl<'a> IntoIterator for &'a ShardSet {
    type Item = &'a Shard;
    type IntoIter = std::slice::Iter<'a, Shard>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl TreeIdDecoder {
    /// Loads the native Huffman tables from a file retaining the NNFP library offsets.
    pub fn from_library(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_library_bytes(&fs::read(path)?)
    }

    /// Loads the native Huffman tables from bytes retaining the NNFP library offsets.
    ///
    /// The four fixed ranges must extend through offset `0x1fdb2`. Truncated data returns
    /// [`Error::Format`].
    pub fn from_library_bytes(data: &[u8]) -> Result<Self> {
        let primary_bytes = data
            .get(0x1f198..0x1f598)
            .ok_or_else(|| Error::Format("native Huffman primary table is truncated".into()))?;
        let fast: [u8; 1_024] = data
            .get(0x1f598..0x1f998)
            .ok_or_else(|| Error::Format("native Huffman fast table is truncated".into()))?
            .try_into()
            .map_err(|_| Error::Format("native Huffman fast table has the wrong size".into()))?;
        let long_code_bytes = data
            .get(0x1f998..0x1fce0)
            .ok_or_else(|| Error::Format("native Huffman long table is truncated".into()))?;
        let long_symbols: [u8; 210] = data
            .get(0x1fce0..0x1fdb2)
            .ok_or_else(|| Error::Format("native Huffman symbol table is truncated".into()))?
            .try_into()
            .map_err(|_| Error::Format("native Huffman symbol table has the wrong size".into()))?;
        let primary = parse_u32_array(primary_bytes, "primary")?;
        let long_codes = parse_u32_array(long_code_bytes, "long")?;
        Ok(Self {
            primary,
            fast,
            long_codes,
            long_symbols,
        })
    }

    /// Expands a leaf's delta-Huffman stream into nondecreasing numeric track IDs.
    ///
    /// Decoding stops at the extended zero terminator. Missing terminators, unknown codes,
    /// truncated symbols, and running-sum overflow are rejected.
    pub fn decode(&self, data: &[u8]) -> Result<Vec<u32>> {
        self.decode_with_capacity(data, 0)
    }

    fn decode_with_capacity(&self, data: &[u8], capacity: usize) -> Result<Vec<u32>> {
        let mut bits = BitReader::new(data);
        let mut values = Vec::with_capacity(capacity);
        let mut total = 0u32;
        loop {
            let available = bits.remaining().min(24);
            if available == 0 {
                return Err(Error::Format("tree ID stream has no terminator".into()));
            }
            let probe = bits.peek(available)? as u32;
            let mut symbol = self.fast[(probe & 1_023) as usize];
            let width;
            if symbol == 45 {
                let found = self
                    .long_codes
                    .iter()
                    .zip(self.long_symbols)
                    .find_map(|(&encoded, candidate)| {
                        let candidate_width = encoded >> 27;
                        let mask = (1u32 << candidate_width) - 1;
                        (candidate_width as usize <= available && ((encoded ^ probe) & mask) == 0)
                            .then_some((candidate, candidate_width as usize))
                    })
                    .ok_or_else(|| Error::Format("tree ID stream has an unknown code".into()))?;
                symbol = found.0;
                width = found.1;
            } else {
                width = (self.primary[symbol as usize] >> 27) as usize;
            }
            if width == 0 || width > available {
                return Err(Error::Format("tree ID stream has a truncated code".into()));
            }
            bits.advance(width);
            let delta = if symbol == 255 {
                let value = decode_extended(&mut bits)?;
                if value == 0 {
                    return Ok(values);
                }
                value
            } else {
                u32::from(symbol)
            };
            total = total
                .checked_add(delta)
                .ok_or_else(|| Error::Format("tree track ID overflows u32".into()))?;
            values.push(total);
        }
    }
}

impl Shard {
    /// Opens one standalone matcher table and assigns its result label.
    ///
    /// The supplied decoder is shared because every shard uses the same static Huffman tables.
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        decoder: Arc<TreeIdDecoder>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            table: Table::open(path)?,
            decoder,
        })
    }

    fn background_mass(&self) -> f32 {
        let reverse_bytes = self
            .table
            .approximate_offset(b"S")
            .saturating_sub(self.table.approximate_offset(b"R"));
        reverse_bytes as f32 * f32::from_bits(0x3f46_a5af)
    }

    /// Decodes every data block and returns the total record count.
    ///
    /// This also checks block boundaries, compression markers, checksums, and prefix-compressed
    /// entries through the underlying table reader.
    pub fn validate(&self) -> Result<usize> {
        self.table.validate_entries(|key, value| match key {
            [b'M', _, _, _] => decode_track(value).map(|_| ()),
            [b'R', _, _, _] => decode_track_partitions(value).map(|_| ()),
            [b'T', _, _, _] => decode_leaf(value, &self.decoder).map(|_| ()),
            [b'M' | b'R' | b'T', ..] => Err(Error::Format(
                "matcher record key has the wrong size".into(),
            )),
            _ => Ok(()),
        })
    }

    /// Looks up one `M` family record by the shard-local numeric track ID.
    ///
    /// Missing keys return `Ok(None)`. Present protobuf fields are decoded into display metadata.
    pub fn track(&self, track_id: u32) -> Result<Option<TrackMetadata>> {
        let key = record_key(b'M', track_id)?;
        self.table
            .get(&key)?
            .map(|value| decode_track(&value))
            .transpose()
    }

    /// Decodes every `M` family record in the shard.
    ///
    /// Returned pairs contain the shard-local 24-bit numeric key and its metadata. Other record
    /// families are skipped.
    pub fn tracks(&self) -> Result<Vec<(u32, TrackMetadata)>> {
        self.table
            .entries()?
            .into_iter()
            .filter_map(|(key, value)| {
                let [b'M', low, middle, high] = key.as_slice() else {
                    return None;
                };
                let track_id = u32::from(*low) | u32::from(*middle) << 8 | u32::from(*high) << 16;
                Some(decode_track(&value).map(|metadata| (track_id, metadata)))
            })
            .collect()
    }

    /// Looks up and decodes one `T` family partition record.
    ///
    /// Codes and track IDs remain in stored occurrence order so duplicate track occurrences are
    /// distinguishable.
    pub fn leaf(&self, leaf_id: u32) -> Result<Option<SearchLeaf>> {
        self.leaf_shared(leaf_id)
            .map(|leaf| leaf.map(|leaf| (*leaf).clone()))
    }

    pub(crate) fn leaf_shared(&self, leaf_id: u32) -> Result<Option<Arc<SearchLeaf>>> {
        let cache_key = CacheKey::Leaf(self.table.id(), leaf_id);
        if let Some(leaf) = self.table.cache().get(cache_key)? {
            return Ok(Some(leaf));
        }
        let key = record_key(b'T', leaf_id)?;
        let Some(leaf) = self
            .table
            .get(&key)?
            .map(|value| decode_leaf(&value, &self.decoder))
            .transpose()?
        else {
            return Ok(None);
        };
        let leaf = Arc::new(leaf);
        let charge = std::mem::size_of::<SearchLeaf>()
            + leaf.codes.capacity() * std::mem::size_of::<[u8; SUBQUANTIZER_COUNT]>()
            + leaf.track_ids.capacity() * std::mem::size_of::<u32>();
        self.table.cache().insert(cache_key, leaf, charge).map(Some)
    }

    /// Decodes a track's `R` family list of partition occurrences.
    ///
    /// Four unsigned ten-bit partition IDs are packed into each five-byte group. Trailing `1023`
    /// values are padding and are removed.
    pub fn track_partitions(&self, track_id: u32) -> Result<Option<Vec<u16>>> {
        self.track_partitions_shared(track_id)
            .map(|partitions| partitions.map(|partitions| (*partitions).clone()))
    }

    fn track_partitions_shared(&self, track_id: u32) -> Result<Option<Arc<Vec<u16>>>> {
        let cache_key = CacheKey::Partitions(self.table.id(), track_id);
        if let Some(partitions) = self.table.cache().get(cache_key)? {
            return Ok(Some(partitions));
        }
        let key = record_key(b'R', track_id)?;
        let Some(partitions) = self
            .table
            .get(&key)?
            .map(|value| decode_track_partitions(&value))
            .transpose()?
        else {
            return Ok(None);
        };
        let partitions = Arc::new(partitions);
        let charge =
            std::mem::size_of::<Vec<u16>>() + partitions.capacity() * std::mem::size_of::<u16>();
        self.table
            .cache()
            .insert(cache_key, partitions, charge)
            .map(Some)
    }

    pub(crate) fn track_partitions_batch(
        &self,
        track_ids: &[u32],
    ) -> Result<HashMap<u32, Vec<u16>>> {
        let keys = track_ids
            .iter()
            .map(|&track_id| record_key(b'R', track_id))
            .collect::<Result<Vec<_>>>()?;
        let values = self.table.get_many_uncached(&keys)?;
        let mut partitions = HashMap::with_capacity(values.len());
        for &track_id in track_ids {
            let key = record_key(b'R', track_id)?;
            if let Some(value) = values.get(&key) {
                partitions.insert(track_id, decode_track_partitions(value)?);
            }
        }
        Ok(partitions)
    }

    /// Resolves one stored occurrence of a track to its partition and 12-byte code.
    ///
    /// `occurrence` indexes the reverse-index sequence rather than unique partitions. Missing
    /// tracks or out-of-range occurrences return `Ok(None)`, while broken cross-record links are
    /// format errors.
    pub fn track_code(&self, track_id: u32, occurrence: usize) -> Result<Option<StoredCode>> {
        let Some(partitions) = self.track_partitions_shared(track_id)? else {
            return Ok(None);
        };
        let Some(&partition) = partitions.get(occurrence) else {
            return Ok(None);
        };
        let ordinal = partitions[..occurrence]
            .iter()
            .filter(|&&candidate| candidate == partition)
            .count();
        let leaf = self.leaf_shared(u32::from(partition))?.ok_or_else(|| {
            Error::Format(format!(
                "reverse index references missing partition {partition}"
            ))
        })?;
        let first = leaf
            .track_ids
            .partition_point(|&candidate| candidate < track_id);
        let code = leaf
            .track_ids
            .get(first + ordinal)
            .filter(|&&candidate| candidate == track_id)
            .and_then(|_| leaf.codes.get(first + ordinal))
            .copied()
            .ok_or_else(|| {
                Error::Format(format!(
                    "reverse index occurrence {occurrence} for track {track_id} has no tree code"
                ))
            })?;
        Ok(Some(StoredCode {
            partition: u32::from(partition),
            code,
        }))
    }

    /// Resolves every reverse-index occurrence for a track in its stored order.
    ///
    /// Repeated assignments to one partition remain repeated and are matched to the corresponding
    /// ordinal occurrence in that partition's candidate leaf.
    pub fn track_codes(&self, track_id: u32) -> Result<Option<Vec<StoredCode>>> {
        self.track_codes_shared(track_id)
            .map(|codes| codes.map(|codes| (*codes).clone()))
    }

    pub(crate) fn track_codes_shared(&self, track_id: u32) -> Result<Option<Arc<Vec<StoredCode>>>> {
        let key = CacheKey::Codes(self.table.id(), track_id);
        if let Some(codes) = self.table.cache().get(key)? {
            return Ok(Some(codes));
        }
        let Some(partitions) = self.track_partitions_shared(track_id)? else {
            return Ok(None);
        };
        let mut leaves = HashMap::new();
        let mut ordinals = HashMap::<u16, usize>::new();
        let mut codes = Vec::with_capacity(partitions.len());
        for &partition in partitions.iter() {
            let leaf = match leaves.entry(partition) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let leaf = self.leaf_shared(u32::from(partition))?.ok_or_else(|| {
                        Error::Format(format!(
                            "reverse index references missing partition {partition}"
                        ))
                    })?;
                    entry.insert(leaf)
                }
            };
            let ordinal = ordinals.entry(partition).or_default();
            let first = leaf
                .track_ids
                .partition_point(|&candidate| candidate < track_id);
            let code = leaf
                .track_ids
                .get(first + *ordinal)
                .filter(|&&candidate| candidate == track_id)
                .and_then(|_| leaf.codes.get(first + *ordinal))
                .copied()
                .ok_or_else(|| {
                    Error::Format(format!(
                        "reverse index occurrence {} for track {track_id} has no tree code",
                        codes.len()
                    ))
                })?;
            *ordinal += 1;
            codes.push(StoredCode {
                partition: u32::from(partition),
                code,
            });
        }
        let codes = Arc::new(codes);
        let charge = std::mem::size_of::<Vec<StoredCode>>()
            + codes.capacity() * std::mem::size_of::<StoredCode>();
        self.table.cache().insert(key, codes, charge).map(Some)
    }
}

fn record_key(family: u8, identifier: u32) -> Result<[u8; 4]> {
    if identifier > 0x00ff_ffff {
        return Err(Error::InvalidInput("record ID exceeds 24 bits".into()));
    }
    Ok([
        family,
        identifier as u8,
        (identifier >> 8) as u8,
        (identifier >> 16) as u8,
    ])
}

fn parse_u32_array<const N: usize>(data: &[u8], name: &str) -> Result<[u32; N]> {
    let expected = N
        .checked_mul(4)
        .ok_or_else(|| Error::Format("native Huffman table size overflows usize".into()))?;
    if data.len() != expected {
        return Err(Error::Format(format!(
            "native Huffman {name} table has {} bytes instead of {expected}",
            data.len()
        )));
    }
    let mut values = [0; N];
    for (value, bytes) in values.iter_mut().zip(data.as_chunks::<4>().0) {
        *value = u32::from_le_bytes(*bytes);
    }
    Ok(values)
}

fn decode_track(data: &[u8]) -> Result<TrackMetadata> {
    let values = fields(data)?;
    Ok(TrackMetadata {
        track_id: optional_string(&values, 1)?,
        catalog_id: optional_string(&values, 2)?,
        title: optional_string(&values, 3)?,
        artist: optional_string(&values, 4)?,
        duration_seconds: optional_fixed64(&values, 6)?.map(f64::from_bits),
        knowledge_graph_id: optional_string(&values, 8)?,
        album: optional_string(&values, 13)?,
        release_year: optional_varint(&values, 14).map(|value| value as u32),
    })
}

fn decode_leaf(data: &[u8], decoder: &TreeIdDecoder) -> Result<SearchLeaf> {
    let values = fields(data)?;
    let code_bytes = one_bytes(&values, 1)?;
    let track_ids = decoder.decode_with_capacity(
        one_bytes(&values, 3)?,
        code_bytes.len() / SUBQUANTIZER_COUNT,
    )?;
    if code_bytes.len() != track_ids.len() * SUBQUANTIZER_COUNT {
        return Err(Error::Format(format!(
            "leaf has {} code bytes for {} tracks",
            code_bytes.len(),
            track_ids.len()
        )));
    }
    let codes = code_bytes.as_chunks::<SUBQUANTIZER_COUNT>().0.to_vec();
    Ok(SearchLeaf { codes, track_ids })
}

fn decode_track_partitions(data: &[u8]) -> Result<Vec<u16>> {
    if !data.len().is_multiple_of(5) {
        return Err(Error::Format(
            "reverse index byte count is not divisible by five".into(),
        ));
    }
    let mut partitions = Vec::with_capacity(data.len() / 5 * 4);
    for group in data.as_chunks::<5>().0 {
        for index in 0..4 {
            let partition = u16::from(group[index]) | u16::from((group[4] >> (2 * index)) & 3) << 8;
            partitions.push(partition);
        }
    }
    while partitions.last() == Some(&1_023) {
        partitions.pop();
    }
    if partitions
        .iter()
        .any(|&partition| usize::from(partition) >= PARTITION_COUNT)
    {
        return Err(Error::Format(
            "reverse index contains an invalid partition".into(),
        ));
    }
    Ok(partitions)
}

fn fields(data: &[u8]) -> Result<Vec<(u32, WireValue<'_>)>> {
    let mut offset = 0;
    let mut values = Vec::new();
    while offset < data.len() {
        let (tag, next) = read_varint(data, offset)?;
        offset = next;
        let number = u32::try_from(tag >> 3)
            .map_err(|_| Error::Format("protobuf field number is too large".into()))?;
        if number == 0 {
            return Err(Error::Format("protobuf field zero is invalid".into()));
        }
        let value = match tag & 7 {
            0 => {
                let (value, next) = read_varint(data, offset)?;
                offset = next;
                WireValue::Varint(value)
            }
            1 => {
                let end = offset
                    .checked_add(8)
                    .ok_or_else(|| Error::Format("fixed64 field range overflows usize".into()))?;
                let value = data
                    .get(offset..end)
                    .ok_or_else(|| Error::Format("truncated fixed64 field".into()))?;
                offset = end;
                WireValue::Fixed64(value)
            }
            2 => {
                let (size, next) = read_varint(data, offset)?;
                offset = next;
                let size = usize::try_from(size)
                    .map_err(|_| Error::Format("protobuf byte field is too large".into()))?;
                let end = offset
                    .checked_add(size)
                    .ok_or_else(|| Error::Format("protobuf byte range overflows usize".into()))?;
                let value = data
                    .get(offset..end)
                    .ok_or_else(|| Error::Format("truncated byte field".into()))?;
                offset = end;
                WireValue::Bytes(value)
            }
            5 => {
                let end = offset
                    .checked_add(4)
                    .ok_or_else(|| Error::Format("fixed32 field range overflows usize".into()))?;
                let value = data
                    .get(offset..end)
                    .ok_or_else(|| Error::Format("truncated fixed32 field".into()))?;
                offset = end;
                WireValue::Fixed32(value)
            }
            wire => {
                return Err(Error::Format(format!(
                    "unsupported protobuf wire type {wire}"
                )));
            }
        };
        values.push((number, value));
    }
    Ok(values)
}

fn read_varint(data: &[u8], mut offset: usize) -> Result<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0;
    while offset < data.len() && shift <= 63 {
        let byte = data[offset];
        offset += 1;
        if shift == 63 && byte > 1 {
            return Err(Error::Format(format!(
                "invalid protobuf varint at byte {start}"
            )));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok((value, offset));
        }
        shift += 7;
    }
    Err(Error::Format(format!(
        "invalid protobuf varint at byte {start}"
    )))
}

fn one_bytes<'a>(values: &[(u32, WireValue<'a>)], number: u32) -> Result<&'a [u8]> {
    let matches = values
        .iter()
        .filter_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Bytes(bytes)) => Some(*bytes),
            _ => None,
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(Error::Format(format!("expected one byte field {number}")));
    }
    Ok(matches[0])
}

fn repeated_bytes<'a>(values: &[(u32, WireValue<'a>)], number: u32) -> Vec<&'a [u8]> {
    values
        .iter()
        .filter_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Bytes(bytes)) => Some(*bytes),
            _ => None,
        })
        .collect()
}

fn one_varint(values: &[(u32, WireValue<'_>)], number: u32) -> Result<u64> {
    let matches = values
        .iter()
        .filter_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Varint(value)) => Some(*value),
            _ => None,
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(Error::Format(format!("expected one varint field {number}")));
    }
    Ok(matches[0])
}

fn optional_varint(values: &[(u32, WireValue<'_>)], number: u32) -> Option<u64> {
    values
        .iter()
        .find_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Varint(value)) => Some(*value),
            _ => None,
        })
}

fn one_f32(values: &[(u32, WireValue<'_>)], number: u32) -> Result<f32> {
    optional_f32(values, number)?
        .ok_or_else(|| Error::Format(format!("expected one fixed32 field {number}")))
}

fn optional_f32(values: &[(u32, WireValue<'_>)], number: u32) -> Result<Option<f32>> {
    let matches = values
        .iter()
        .filter_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Fixed32(value)) => Some(*value),
            _ => None,
        })
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(Error::Format(format!(
            "expected at most one fixed32 field {number}"
        )));
    }
    matches
        .first()
        .map(|value| {
            let bytes = (*value)
                .try_into()
                .map_err(|_| Error::Format(format!("fixed32 field {number} has the wrong size")))?;
            Ok(f32::from_le_bytes(bytes))
        })
        .transpose()
}

fn optional_fixed64(values: &[(u32, WireValue<'_>)], number: u32) -> Result<Option<u64>> {
    values
        .iter()
        .find_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Fixed64(value)) => Some(*value),
            _ => None,
        })
        .map(|value| {
            let bytes = value
                .try_into()
                .map_err(|_| Error::Format(format!("fixed64 field {number} has the wrong size")))?;
            Ok(u64::from_le_bytes(bytes))
        })
        .transpose()
}

fn optional_string(values: &[(u32, WireValue<'_>)], number: u32) -> Result<String> {
    let Some(value) = values
        .iter()
        .find_map(|(field, value)| match (*field == number, value) {
            (true, WireValue::Bytes(value)) => Some(*value),
            _ => None,
        })
    else {
        return Ok(String::new());
    };
    String::from_utf8(value.to_vec())
        .map_err(|_| Error::Format(format!("field {number} is not UTF-8")))
}

fn parse_packed_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    if data.len() != count * 4 {
        return Err(Error::Format(format!(
            "packed float field has {} bytes instead of {}",
            data.len(),
            count * 4
        )));
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_position: usize,
    buffer: u64,
    buffered: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_position: 0,
            buffer: 0,
            buffered: 0,
        }
    }

    fn remaining(&self) -> usize {
        (self.data.len() - self.byte_position) * 8 + self.buffered
    }

    fn peek(&mut self, count: usize) -> Result<u64> {
        if count > self.remaining() {
            return Err(Error::Format("bit read extends beyond the input".into()));
        }
        while self.buffered < count {
            self.buffer |= u64::from(self.data[self.byte_position]) << self.buffered;
            self.byte_position += 1;
            self.buffered += 8;
        }
        let mask = if count == 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        Ok(self.buffer & mask)
    }

    fn read(&mut self, count: usize) -> Result<u64> {
        let value = self.peek(count)?;
        self.buffer >>= count;
        self.buffered -= count;
        Ok(value)
    }

    fn advance(&mut self, count: usize) {
        self.buffer >>= count;
        self.buffered -= count;
    }
}

fn decode_extended(bits: &mut BitReader<'_>) -> Result<u32> {
    let mut ones = 0usize;
    while bits.remaining() != 0 && bits.read(1)? == 1 {
        ones += 1;
    }
    let groups = ones
        .checked_add(1)
        .ok_or_else(|| Error::Format("extended tree ID width overflows usize".into()))?;
    if groups > 8 {
        return Err(Error::Format("extended tree ID is wider than u32".into()));
    }
    let width = groups * 4;
    let raw = u32::try_from(bits.read(width)?)
        .map_err(|_| Error::Format("extended tree ID does not fit u32".into()))?;
    let prefix = (1..groups).try_fold(0u32, |value, index| {
        value
            .checked_add(1u32 << (4 * index))
            .ok_or_else(|| Error::Format("extended tree ID overflows u32".into()))
    })?;
    raw.checked_add(prefix)
        .ok_or_else(|| Error::Format("extended tree ID overflows u32".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_recovered_matcher_config() {
        let config = MatcherConfig::from_file("../../assets/v3_config_tah.pb").unwrap();
        assert_eq!(config.partition_count, PARTITION_COUNT);
        assert_eq!(config.probe_count, PROBE_COUNT);
        assert_eq!(config.scorer, ScorerConfig::default());
        assert_eq!(config.acceptance_threshold.to_bits(), 0);
        assert_eq!(config.codebook.len(), 12 * 256 * 8);
        assert_eq!(config.centroids.len(), 1_000 * 96);

        let expected_codebook = include_bytes!("../tests/fixtures/ah-codebook.f32le");
        for (&actual, expected) in config
            .codebook
            .iter()
            .zip(expected_codebook.as_chunks::<4>().0)
        {
            assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
        }
        let expected_centroids = include_bytes!("../tests/fixtures/partition-centroids.f32le");
        for (&actual, expected) in config
            .centroids
            .iter()
            .zip(expected_centroids.as_chunks::<4>().0)
        {
            assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
        }
    }

    #[test]
    #[ignore = "requires NOWPLAYING_MATCHER_PATH"]
    fn decodes_real_tree_and_metadata_records() {
        let decoder =
            Arc::new(TreeIdDecoder::from_library("../../assets/nnfp_v3.weights").unwrap());
        let matcher_path = std::env::var("NOWPLAYING_MATCHER_PATH").unwrap();
        let shard = Shard::open(".core", matcher_path, decoder).unwrap();
        let track = shard.track(0).unwrap().unwrap();
        assert!(!track.track_id.is_empty());
        assert!(!track.title.is_empty());
        let tracks = shard.tracks().unwrap();
        assert!(
            tracks
                .iter()
                .any(|(track_id, candidate)| { *track_id == 0 && candidate == &track })
        );
        let leaf = shard.leaf(0).unwrap().unwrap();
        assert_eq!(leaf.codes.len(), leaf.track_ids.len());
        assert!(!leaf.codes.is_empty());
        let partitions = shard.track_partitions(0).unwrap().unwrap();
        assert!(!partitions.is_empty());
        assert!(
            partitions
                .iter()
                .all(|&partition| usize::from(partition) < PARTITION_COUNT)
        );
        let codes = shard.track_codes(0).unwrap().unwrap();
        assert_eq!(codes.len(), partitions.len());
        assert_eq!(shard.track_code(0, 0).unwrap(), codes.first().copied());
    }

    #[test]
    #[ignore = "requires NOWPLAYING_ARTIFACTS_PATH"]
    fn derives_native_background_mass_from_table_offsets() {
        let artifacts = std::env::var("NOWPLAYING_ARTIFACTS_PATH").unwrap();
        let decoder =
            Arc::new(TreeIdDecoder::from_library("../../assets/nnfp_v3.weights").unwrap());
        let core = Shard::open(
            "matcher_tah.leveldb",
            format!("{artifacts}/etc/ambient/matcher_tah.leveldb"),
            decoder.clone(),
        )
        .unwrap();
        let us = Shard::open(
            "US0d8db236a287d657caca26073f6165d8",
            format!("{artifacts}/etc/ambient/US0d8db236a287d657caca26073f6165d8"),
            decoder,
        )
        .unwrap();
        assert_eq!(core.background_mass().to_bits(), 3_807_674.3_f32.to_bits());
        assert_eq!(us.background_mass().to_bits(), 542_046.94_f32.to_bits());
        assert_eq!(
            (core.background_mass() + us.background_mass()).to_bits(),
            4_349_721.0f32.to_bits()
        );
    }

    #[test]
    fn oversized_extended_tree_id_is_rejected() {
        let mut primary = [0; 256];
        primary[255] = 1 << 27;
        let decoder = TreeIdDecoder {
            primary,
            fast: [255; 1_024],
            long_codes: [0; 210],
            long_symbols: [0; 210],
        };

        assert!(decoder.decode(&[0xff, 0x03, 0, 0, 0, 0, 0]).is_err());
    }
}
