//! Reader for Google's standalone LevelDB matcher tables.
//!
//! Matcher shards are single table files rather than LevelDB directories. They contain the
//! standard footer, an empty metaindex, an index block, and directly addressed data blocks. This
//! reader supports the uncompressed block form used by the public catalog and verifies every
//! block's masked CRC32C before returning its contents.

use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{Error, Result};

const MAGIC: u64 = 0xdb4775248b80fb57;
pub(crate) const DEFAULT_CACHE_BYTES: usize = 16 * 1024 * 1024;
const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MADV_DONTNEED: i32 = 4;
#[cfg(any(not(target_arch = "aarch64"), test))]
const CRC32C_TABLE: [u32; 256] = crc32c_table();
static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(1);
type Entry = (Vec<u8>, Vec<u8>);
type BlockEntries = Arc<Vec<Entry>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum CacheKey {
    Block(u64, usize),
    Leaf(u64, u32),
    Partitions(u64, u32),
    Codes(u64, u32),
}

struct CacheEntry {
    value: Arc<dyn Any + Send + Sync>,
    charge: usize,
    touched: u64,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<CacheKey, CacheEntry>,
    lru: BTreeSet<(u64, CacheKey)>,
    used: usize,
    clock: u64,
}

pub(crate) struct DecodedCache {
    limit: Option<usize>,
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for DecodedCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodedCache")
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl DecodedCache {
    pub(crate) fn new(limit: Option<usize>) -> Self {
        Self {
            limit,
            state: Mutex::new(CacheState::default()),
        }
    }

    pub(crate) fn get<T>(&self, key: CacheKey) -> Result<Option<Arc<T>>>
    where
        T: Any + Send + Sync,
    {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Format("decoded cache is poisoned".into()))?;
        state.clock = state.clock.wrapping_add(1);
        let touched = state.clock;
        let Some((previous, value)) = state.entries.get_mut(&key).map(|entry| {
            let previous = entry.touched;
            entry.touched = touched;
            (previous, Arc::clone(&entry.value))
        }) else {
            return Ok(None);
        };
        state.lru.remove(&(previous, key));
        state.lru.insert((touched, key));
        Arc::downcast(value)
            .map(Some)
            .map_err(|_| Error::Format("decoded cache type mismatch".into()))
    }

    pub(crate) fn insert<T>(&self, key: CacheKey, value: Arc<T>, charge: usize) -> Result<Arc<T>>
    where
        T: Any + Send + Sync,
    {
        if let Some(value) = self.get(key)? {
            return Ok(value);
        }
        if self.limit.is_some_and(|limit| charge > limit) {
            return Ok(value);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Format("decoded cache is poisoned".into()))?;
        if let Some(existing) = state.entries.get(&key) {
            return Arc::downcast(Arc::clone(&existing.value))
                .map_err(|_| Error::Format("decoded cache type mismatch".into()));
        }
        if let Some(limit) = self.limit {
            while state.used.saturating_add(charge) > limit {
                let Some(&(touched, oldest)) = state.lru.first() else {
                    break;
                };
                state.lru.remove(&(touched, oldest));
                if let Some(removed) = state.entries.remove(&oldest) {
                    state.used = state.used.saturating_sub(removed.charge);
                }
            }
        }
        state.clock = state.clock.wrapping_add(1);
        let touched = state.clock;
        state.used = state.used.saturating_add(charge);
        state.lru.insert((touched, key));
        state.entries.insert(
            key,
            CacheEntry {
                value: Arc::clone(&value) as Arc<dyn Any + Send + Sync>,
                charge,
                touched,
            },
        );
        Ok(value)
    }
}

#[derive(Debug)]
struct MappedFile {
    address: usize,
    len: usize,
}

impl MappedFile {
    fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| Error::Format("table length does not fit usize".into()))?;
        if len == 0 {
            return Err(Error::Format("table is empty".into()));
        }
        #[expect(unsafe_code, reason = "read-only file mapping")]
        // SAFETY: `file` is valid for `len` bytes for this call. Production tables are either on
        // the read-only product image or beneath app-private 0700 directories. Catalog updates
        // write a separate partial file and rename or unlink it, so they never truncate this inode.
        // A rare successful null mapping is released with the same address and length. An in-place
        // updater would break the file-size invariant and could make later reads raise SIGBUS.
        let address = unsafe {
            let address = mmap(
                ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            );
            if address.is_null() {
                munmap(address, len);
            }
            address
        } as usize;
        if address == usize::MAX {
            return Err(std::io::Error::last_os_error().into());
        }
        if address == 0 {
            return Err(Error::Format("mmap returned a null address".into()));
        }
        Ok(Self { address, len })
    }

    fn as_slice(&self) -> &[u8] {
        #[expect(unsafe_code, reason = "mapping remains valid for self lifetime")]
        // SAFETY: `address` is a non-null mapping of exactly `len` readable bytes. The mapping is
        // immutable and remains owned by `self`, so the returned borrow cannot outlive `munmap`.
        unsafe {
            std::slice::from_raw_parts(self.address as *const u8, self.len)
        }
    }

    fn discard_range(&self, offset: usize, len: usize) {
        let page_size = platform_page_size();
        if page_size == 0 || offset >= self.len {
            return;
        }
        let start = offset / page_size * page_size;
        let end = offset
            .saturating_add(len)
            .min(self.len)
            .div_ceil(page_size)
            .saturating_mul(page_size)
            .min(self.len);
        if end <= start {
            return;
        }
        #[expect(unsafe_code, reason = "discard clean pages from owned mapping")]
        // SAFETY: the mapping address and `start` are page aligned, and the saturated range stays
        // within the live mapping. `MADV_DONTNEED` preserves the mapping and all Rust references.
        unsafe {
            madvise(
                (self.address + start) as *mut c_void,
                end - start,
                MADV_DONTNEED,
            );
        }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[expect(unsafe_code, reason = "release owned file mapping")]
        // SAFETY: this object owns the live mapping at `address` with exactly `len` bytes. Drop
        // runs once, and borrows returned by `as_slice` cannot outlive this object.
        unsafe {
            munmap(self.address as *mut c_void, self.len);
        }
    }
}

#[derive(Debug)]
enum TableData {
    Mapped(MappedFile),
    Owned(Box<[u8]>),
}

impl TableData {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(mapped) => mapped.as_slice(),
            Self::Owned(data) => data,
        }
    }

    fn discard_range(&self, offset: usize, len: usize) {
        if let Self::Mapped(mapped) = self {
            mapped.discard_range(offset, len);
        }
    }

    fn discard_all(&self) {
        if let Self::Mapped(mapped) = self {
            mapped.discard_range(0, mapped.len);
        }
    }
}

#[expect(unsafe_code, reason = "query platform page size")]
fn platform_page_size() -> usize {
    // SAFETY: `getpagesize` takes no arguments and has no caller-side preconditions.
    let page_size = unsafe { getpagesize() };
    usize::try_from(page_size).unwrap_or(0)
}

// SAFETY: these declarations match Bionic's signatures and are only called through the checked
// mapping, advice, page-size, and unmapping sites above.
#[expect(unsafe_code, reason = "platform mmap interface")]
unsafe extern "C" {
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: i32,
        flags: i32,
        file_descriptor: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> i32;
    fn madvise(address: *mut c_void, length: usize, advice: i32) -> i32;
    fn getpagesize() -> i32;
}

#[derive(Debug, Clone, Copy)]
struct BlockHandle {
    offset: usize,
    size: usize,
}

#[derive(Debug, Clone)]
struct IndexEntry {
    separator: Vec<u8>,
    handle: BlockHandle,
}

/// An immutable standalone LevelDB table backed by a read-only file mapping.
///
/// Construction parses the footer and index block. Data blocks are decoded on lookup so malformed
/// keys, restart arrays, trailers, or checksums remain errors even after the table has opened.
#[derive(Debug, Clone)]
pub struct Table {
    id: u64,
    data: Arc<TableData>,
    index: Vec<IndexEntry>,
    cache: Arc<DecodedCache>,
}

impl Table {
    /// Reads a standalone LevelDB table from disk and parses its block index.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_data(TableData::Mapped(MappedFile::open(path)?))
    }

    /// Parses a standalone LevelDB table from owned bytes.
    ///
    /// The production tables are uncompressed. Any other compression marker, a wrong footer
    /// magic value, an invalid index handle, or a bad index checksum returns [`Error::Format`].
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_data(TableData::Owned(data.into_boxed_slice()))
    }

    fn from_data(data: TableData) -> Result<Self> {
        let data = Arc::new(data);
        let bytes = data.as_slice();
        if bytes.len() < 48 {
            return Err(Error::Format("table is shorter than its footer".into()));
        }
        let magic = parse_u64(&bytes[bytes.len() - 8..], "table magic")?;
        if magic != MAGIC {
            return Err(Error::Format(format!("wrong LevelDB magic 0x{magic:016x}")));
        }
        let footer = &bytes[bytes.len() - 48..bytes.len() - 8];
        let (_, offset) = read_handle(footer, 0)?;
        let (index_handle, _) = read_handle(footer, offset)?;
        let index_block = read_block(bytes, index_handle)?;
        let index = decode_entries(index_block)?
            .into_iter()
            .map(|(separator, encoded_handle)| {
                let (handle, end) = read_handle(&encoded_handle, 0)?;
                if end != encoded_handle.len() {
                    return Err(Error::Format("index handle has trailing bytes".into()));
                }
                Ok(IndexEntry { separator, handle })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            id: NEXT_TABLE_ID.fetch_add(1, Ordering::Relaxed),
            data,
            index,
            cache: Arc::new(DecodedCache::new(Some(DEFAULT_CACHE_BYTES))),
        })
    }

    /// Returns the number of data-block handles named by the table index.
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Retrieves an exact bytewise key from the data block selected by the index.
    ///
    /// Values are copied out of the table. A missing key returns `Ok(None)`, while malformed block
    /// data or a checksum mismatch returns [`Error::Format`].
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let block_index = self
            .index
            .partition_point(|entry| entry.separator.as_slice() < key);
        let Some(entry) = self.index.get(block_index) else {
            return Ok(None);
        };
        let entries = self.block_entries(block_index, entry)?;
        Ok(entries
            .iter()
            .find_map(|(entry_key, value)| (entry_key == key).then(|| value.clone())))
    }

    pub(crate) fn get_many_uncached(&self, keys: &[[u8; 4]]) -> Result<HashMap<[u8; 4], Vec<u8>>> {
        let mut indexed = keys
            .iter()
            .copied()
            .filter_map(|key| {
                let block_index = self
                    .index
                    .partition_point(|entry| entry.separator.as_slice() < key.as_slice());
                self.index.get(block_index).map(|_| (block_index, key))
            })
            .collect::<Vec<_>>();
        indexed.sort_unstable();
        let mut values = HashMap::with_capacity(indexed.len());
        let mut start = 0;
        while start < indexed.len() {
            let block_index = indexed[start].0;
            let end = indexed[start..].partition_point(|&(candidate, _)| candidate == block_index)
                + start;
            let entry = &self.index[block_index];
            find_entries(
                read_block(self.data.as_slice(), entry.handle)?,
                &indexed[start..end],
                &mut values,
            )?;
            self.discard_block(entry.handle);
            start = end;
        }
        Ok(values)
    }

    /// Decodes every key and value in table order.
    ///
    /// This materializes the entire table and is intended for validation and inventory work rather
    /// than recognition's point lookups.
    pub fn entries(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut entries = Vec::new();
        for (block_index, index) in self.index.iter().enumerate() {
            entries.extend(self.block_entries(block_index, index)?.iter().cloned());
        }
        Ok(entries)
    }

    /// Checks every indexed data block and returns the number of decoded records.
    ///
    /// Validation covers block ranges, the uncompressed marker, masked CRC32C, prefix-compressed
    /// entries, and restart-array boundaries.
    pub fn validate(&self) -> Result<usize> {
        self.validate_entries(|_, _| Ok(()))
    }

    pub(crate) fn validate_entries(
        &self,
        mut validate: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<usize> {
        let mut count = 0usize;
        for index in &self.index {
            let entries = decode_entries(read_block(self.data.as_slice(), index.handle)?)?;
            for (key, value) in &entries {
                validate(key, value)?;
            }
            count = count
                .checked_add(entries.len())
                .ok_or_else(|| Error::Format("table record count overflows usize".into()))?;
            self.discard_block(index.handle);
        }
        Ok(count)
    }

    pub(crate) fn install_cache(&mut self, cache: Arc<DecodedCache>) {
        self.cache = cache;
    }

    pub(crate) fn cache(&self) -> &Arc<DecodedCache> {
        &self.cache
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn approximate_offset(&self, key: &[u8]) -> usize {
        let block_index = self
            .index
            .partition_point(|entry| entry.separator.as_slice() < key);
        self.index
            .get(block_index)
            .map_or(self.data.as_slice().len(), |entry| entry.handle.offset)
    }

    pub(crate) fn discard_mapped_pages(&self) {
        self.data.discard_all();
    }

    fn block_entries(&self, block_index: usize, index: &IndexEntry) -> Result<BlockEntries> {
        let key = CacheKey::Block(self.id, block_index);
        if let Some(entries) = self.cache().get(key)? {
            return Ok(entries);
        }
        let decoded = Arc::new(decode_entries(read_block(
            self.data.as_slice(),
            index.handle,
        )?)?);
        self.discard_block(index.handle);
        let charge = decoded.capacity() * std::mem::size_of::<Entry>()
            + decoded
                .iter()
                .map(|(key, value)| key.capacity() + value.capacity())
                .sum::<usize>();
        self.cache().insert(key, decoded, charge)
    }

    fn discard_block(&self, handle: BlockHandle) {
        self.data
            .discard_range(handle.offset, handle.size.saturating_add(5));
    }
}

fn read_handle(data: &[u8], offset: usize) -> Result<(BlockHandle, usize)> {
    let (block_offset, offset) = read_varint(data, offset)?;
    let (block_size, offset) = read_varint(data, offset)?;
    let offset_value = usize::try_from(block_offset)
        .map_err(|_| Error::Format("block offset does not fit usize".into()))?;
    let size_value = usize::try_from(block_size)
        .map_err(|_| Error::Format("block size does not fit usize".into()))?;
    Ok((
        BlockHandle {
            offset: offset_value,
            size: size_value,
        },
        offset,
    ))
}

fn read_varint(data: &[u8], mut offset: usize) -> Result<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0;
    while offset < data.len() && shift <= 63 {
        let byte = data[offset];
        offset += 1;
        if shift == 63 && byte > 1 {
            return Err(Error::Format(format!("invalid varint at byte {start}")));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok((value, offset));
        }
        shift += 7;
    }
    Err(Error::Format(format!("invalid varint at byte {start}")))
}

fn read_block(data: &[u8], handle: BlockHandle) -> Result<&[u8]> {
    let end = handle
        .offset
        .checked_add(handle.size)
        .ok_or_else(|| Error::Format("block range overflows usize".into()))?;
    let trailer_end = end
        .checked_add(5)
        .ok_or_else(|| Error::Format("block trailer range overflows usize".into()))?;
    if trailer_end > data.len() {
        return Err(Error::Format(format!(
            "block at {} extends beyond the table",
            handle.offset
        )));
    }
    let contents = &data[handle.offset..end];
    let compression = data[end];
    if compression != 0 {
        return Err(Error::Format(format!(
            "unsupported compression type {compression} at {}",
            handle.offset
        )));
    }
    let stored_crc = parse_u32(&data[end + 1..trailer_end], "block checksum")?;
    let computed_crc = mask_crc(crc32c_block(contents, compression));
    if stored_crc != computed_crc {
        return Err(Error::Format(format!(
            "CRC mismatch at block {}",
            handle.offset
        )));
    }
    Ok(contents)
}

fn decode_entries(contents: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    if contents.len() < 4 {
        return Err(Error::Format(
            "block is shorter than its restart count".into(),
        ));
    }
    let restart_count = parse_u32(&contents[contents.len() - 4..], "restart count")? as usize;
    let restart_bytes = restart_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| Error::Format("restart array size overflows usize".into()))?;
    let restart_start = contents
        .len()
        .checked_sub(restart_bytes)
        .ok_or_else(|| Error::Format("restart array extends before the block".into()))?;
    let mut offset = 0;
    let mut previous_key = Vec::new();
    let mut entries = Vec::new();
    while offset < restart_start {
        let (shared, next) = read_varint(contents, offset)?;
        let (nonshared, next) = read_varint(contents, next)?;
        let (value_size, next) = read_varint(contents, next)?;
        offset = next;
        let shared = usize::try_from(shared)
            .map_err(|_| Error::Format("shared key size is too large".into()))?;
        let nonshared = usize::try_from(nonshared)
            .map_err(|_| Error::Format("key size is too large".into()))?;
        let value_size = usize::try_from(value_size)
            .map_err(|_| Error::Format("value size is too large".into()))?;
        let key_end = offset
            .checked_add(nonshared)
            .ok_or_else(|| Error::Format("entry key range overflows usize".into()))?;
        let value_end = key_end
            .checked_add(value_size)
            .ok_or_else(|| Error::Format("entry value range overflows usize".into()))?;
        if shared > previous_key.len() || value_end > restart_start {
            return Err(Error::Format("entry extends outside its block".into()));
        }
        let mut key = previous_key[..shared].to_vec();
        key.extend_from_slice(&contents[offset..key_end]);
        let value = contents[key_end..value_end].to_vec();
        offset = value_end;
        previous_key = key.clone();
        entries.push((key, value));
    }
    if offset != restart_start {
        return Err(Error::Format(
            "entry stream misses the restart array".into(),
        ));
    }
    Ok(entries)
}

fn find_entries(
    contents: &[u8],
    targets: &[(usize, [u8; 4])],
    values: &mut HashMap<[u8; 4], Vec<u8>>,
) -> Result<()> {
    if contents.len() < 4 {
        return Err(Error::Format(
            "block is shorter than its restart count".into(),
        ));
    }
    let restart_count = parse_u32(&contents[contents.len() - 4..], "restart count")? as usize;
    let restart_bytes = restart_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| Error::Format("restart array size overflows usize".into()))?;
    let restart_start = contents
        .len()
        .checked_sub(restart_bytes)
        .ok_or_else(|| Error::Format("restart array extends before the block".into()))?;
    let mut offset = 0;
    let mut previous_key = Vec::new();
    let mut target_index = 0;
    while offset < restart_start && target_index < targets.len() {
        let (shared, next) = read_varint(contents, offset)?;
        let (nonshared, next) = read_varint(contents, next)?;
        let (value_size, next) = read_varint(contents, next)?;
        offset = next;
        let shared = usize::try_from(shared)
            .map_err(|_| Error::Format("shared key size is too large".into()))?;
        let nonshared = usize::try_from(nonshared)
            .map_err(|_| Error::Format("key size is too large".into()))?;
        let value_size = usize::try_from(value_size)
            .map_err(|_| Error::Format("value size is too large".into()))?;
        let key_end = offset
            .checked_add(nonshared)
            .ok_or_else(|| Error::Format("entry key range overflows usize".into()))?;
        let value_end = key_end
            .checked_add(value_size)
            .ok_or_else(|| Error::Format("entry value range overflows usize".into()))?;
        if shared > previous_key.len() || value_end > restart_start {
            return Err(Error::Format("entry extends outside its block".into()));
        }
        previous_key.truncate(shared);
        previous_key.extend_from_slice(&contents[offset..key_end]);
        while target_index < targets.len()
            && targets[target_index].1.as_slice() < previous_key.as_slice()
        {
            target_index += 1;
        }
        if target_index < targets.len()
            && targets[target_index].1.as_slice() == previous_key.as_slice()
        {
            let target = targets[target_index].1;
            values.insert(target, contents[key_end..value_end].to_vec());
            target_index += 1;
        }
        offset = value_end;
    }
    if offset > restart_start {
        return Err(Error::Format(
            "entry stream extends into the restart array".into(),
        ));
    }
    Ok(())
}

fn parse_u32(data: &[u8], name: &str) -> Result<u32> {
    let bytes = data
        .try_into()
        .map_err(|_| Error::Format(format!("{name} has {} bytes instead of 4", data.len())))?;
    Ok(u32::from_le_bytes(bytes))
}

fn parse_u64(data: &[u8], name: &str) -> Result<u64> {
    let bytes = data
        .try_into()
        .map_err(|_| Error::Format(format!("{name} has {} bytes instead of 8", data.len())))?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
fn crc32c(data: &[u8]) -> u32 {
    crc32c_iter(data.iter().copied())
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn crc32c_iter(data: impl Iterator<Item = u8>) -> u32 {
    let mut crc = u32::MAX;
    for byte in data {
        crc = CRC32C_TABLE[((crc ^ u32::from(byte)) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ u32::MAX
}

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_code)]
fn crc32c_block(contents: &[u8], compression: u8) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};

    let (chunks, tail) = contents.as_chunks::<8>();
    let mut crc = u32::MAX;
    // SAFETY: this function is compiled only for AArch64, and the CRC intrinsics operate entirely
    // on copied integer values without dereferencing pointers or retaining state.
    unsafe {
        for chunk in chunks {
            crc = __crc32cd(crc, u64::from_le_bytes(*chunk));
        }
        for &byte in tail {
            crc = __crc32cb(crc, byte);
        }
        !__crc32cb(crc, compression)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn crc32c_block(contents: &[u8], compression: u8) -> u32 {
    crc32c_iter(contents.iter().copied().chain(std::iter::once(compression)))
}

#[cfg(any(not(target_arch = "aarch64"), test))]
const fn crc32c_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut index = 0;
    while index < table.len() {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = (value >> 1) ^ if value & 1 == 1 { 0x82f63b78 } else { 0 };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

fn mask_crc(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires NOWPLAYING_MATCHER_PATH"]
    fn validates_the_core_table() {
        let matcher_path = std::env::var("NOWPLAYING_MATCHER_PATH").unwrap();
        let table = Table::open(matcher_path).unwrap();
        assert_eq!(table.block_count(), 2_637);
        assert_eq!(table.validate().unwrap(), 33_000);
        assert!(table.get(&[b'M', 0, 0, 0]).unwrap().is_some());
    }

    #[test]
    fn crc_matches_the_leveldb_fixture() {
        assert_eq!(mask_crc(crc32c(b"abc")), 0x21f1576e);
    }

    #[test]
    fn bounded_cache_evicts_the_oldest_value() {
        let cache = DecodedCache::new(Some(8));
        cache
            .insert(CacheKey::Block(1, 0), Arc::new(vec![1u8; 8]), 8)
            .unwrap();
        cache
            .insert(CacheKey::Block(1, 1), Arc::new(vec![2u8; 8]), 8)
            .unwrap();
        assert!(
            cache
                .get::<Vec<u8>>(CacheKey::Block(1, 0))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            cache
                .get::<Vec<u8>>(CacheKey::Block(1, 1))
                .unwrap()
                .unwrap()
                .as_slice(),
            [2; 8]
        );
    }

    #[test]
    fn arbitrary_table_bytes_do_not_panic() {
        let mut state = 0x1234_5678u32;
        for len in 0..4_096 {
            let mut data = vec![0; len];
            for byte in &mut data {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = state as u8;
            }
            let _ = Table::from_bytes(data);
        }
    }
}
