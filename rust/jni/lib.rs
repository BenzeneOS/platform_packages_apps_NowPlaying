//! JNI boundary for the privileged Now Playing service.
//!
//! The bridge owns one process-wide recognizer behind a mutex. It converts Rust errors and caught
//! panics into `RuntimeException`, leaving model parsing and recognition in `nowplaying_core`.

use std::any::Any;
use std::error;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use jni::JNIEnv;
use jni::objects::{JDoubleArray, JObject, JObjectArray, JShortArray, JString};
use jni::sys::{JNI_TRUE, jboolean, jint, jstring};
use nowplaying_core::embedder::Embedder;
use nowplaying_core::index::{MatcherConfig, Shard, ShardSet, TreeIdDecoder};
use nowplaying_core::music_detector::MusicDetector;
use nowplaying_core::recognize::{RecognitionPolicy, Recognizer};
use nowplaying_core::search::PreviousMatch;

#[derive(Debug)]
struct NativeError(String);

impl NativeError {
    fn from_display(error: impl fmt::Display) -> Self {
        Self(error.to_string())
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for NativeError {}

impl From<jni::errors::Error> for NativeError {
    fn from(error: jni::errors::Error) -> Self {
        Self::from_display(error)
    }
}

impl From<nowplaying_core::Error> for NativeError {
    fn from(error: nowplaying_core::Error) -> Self {
        Self::from_display(error)
    }
}

type Result<T> = std::result::Result<T, NativeError>;

struct NativeRecognizerState {
    recognizer: Recognizer,
    config: MatcherConfig,
    decoder: Arc<TreeIdDecoder>,
}

static RECOGNIZER: OnceLock<Mutex<Option<NativeRecognizerState>>> = OnceLock::new();
static LAST_TIMINGS: OnceLock<Mutex<String>> = OnceLock::new();

#[expect(unsafe_code, reason = "Linux thread affinity")]
mod affinity {
    use std::ffi::c_int;
    use std::io;
    use std::mem::size_of;

    use super::{NativeError, Result};

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CpuSet([u64; 16]);

    // SAFETY: these declarations match Bionic's LP64 affinity signatures. `CpuSet` is the same
    // 1024-bit array of 64-bit words as Bionic's `cpu_set_t`, and every call supplies its size.
    unsafe extern "C" {
        fn sched_getaffinity(pid: c_int, cpusetsize: usize, mask: *mut CpuSet) -> c_int;
        fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: *const CpuSet) -> c_int;
    }

    pub(super) struct Guard {
        previous: CpuSet,
    }

    fn current() -> io::Result<CpuSet> {
        let mut mask = CpuSet([0; 16]);
        // SAFETY: `mask` is writable, aligned, and valid for the supplied object size. The
        // successful call initializes the mask before it is returned.
        if unsafe { sched_getaffinity(0, size_of::<CpuSet>(), &mut mask) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(mask)
    }

    fn set(mask: &CpuSet) -> io::Result<()> {
        // SAFETY: `mask` is fully initialized, aligned, and valid for the supplied object size.
        if unsafe { sched_setaffinity(0, size_of::<CpuSet>(), mask) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    impl Guard {
        pub(super) fn pin_small_cores(enabled: bool) -> Result<Option<Self>> {
            if !enabled {
                return Ok(None);
            }
            let previous = current().map_err(NativeError::from_display)?;
            let mut small_cores = CpuSet([0; 16]);
            small_cores.0[0] = 0xf;
            set(&small_cores).map_err(NativeError::from_display)?;
            Ok(Some(Self { previous }))
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = set(&self.previous);
        }
    }
}

fn recognizer() -> &'static Mutex<Option<NativeRecognizerState>> {
    RECOGNIZER.get_or_init(|| Mutex::new(None))
}

fn last_timings() -> &'static Mutex<String> {
    LAST_TIMINGS.get_or_init(|| Mutex::new(String::new()))
}

fn read_string(env: &mut JNIEnv<'_>, value: &JString<'_>) -> Result<String> {
    env.get_string(value).map(Into::into).map_err(Into::into)
}

fn read_paths(env: &mut JNIEnv<'_>, paths: &JObjectArray<'_>) -> Result<Vec<String>> {
    let count = env.get_array_length(paths)?;
    let mut result = Vec::with_capacity(count as usize);
    for index in 0..count {
        let path = JString::from(env.get_object_array_element(paths, index)?);
        result.push(read_string(env, &path)?);
    }
    Ok(result)
}

fn open_shards(shard_paths: &[String], decoder: &Arc<TreeIdDecoder>) -> Result<Vec<Shard>> {
    shard_paths
        .iter()
        .map(|shard_path| {
            let shard_name = Path::new(shard_path)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| NativeError("shard path has no UTF-8 file name".into()))?;
            Shard::open(shard_name, shard_path, Arc::clone(decoder)).map_err(Into::into)
        })
        .collect()
}

fn initialize(weights_path: &str, config_path: &str, shard_paths: &[String]) -> Result<()> {
    let slot = recognizer()
        .lock()
        .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
    if slot.is_some() {
        return Ok(());
    }
    drop(slot);
    let config = MatcherConfig::from_file(config_path)?;
    let decoder = Arc::new(TreeIdDecoder::from_library(weights_path)?);
    let shards = open_shards(shard_paths, &decoder)?;
    let built_recognizer = Recognizer::new(
        Embedder::from_library(weights_path)?,
        MusicDetector::from_library(weights_path)?,
        ShardSet::new(config.clone(), shards),
        RecognitionPolicy::default(),
    )?;
    let state = NativeRecognizerState {
        recognizer: built_recognizer,
        config,
        decoder,
    };
    let mut slot = recognizer()
        .lock()
        .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
    if slot.is_none() {
        *slot = Some(state);
    }
    Ok(())
}

fn reload(shard_paths: &[String]) -> Result<()> {
    let (config, decoder) = {
        let slot = recognizer()
            .lock()
            .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
        let state = slot
            .as_ref()
            .ok_or_else(|| NativeError("recognizer is not initialized".into()))?;
        (state.config.clone(), Arc::clone(&state.decoder))
    };
    let shards = open_shards(shard_paths, &decoder)?;
    let mut slot = recognizer()
        .lock()
        .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
    let state = slot
        .as_mut()
        .ok_or_else(|| NativeError("recognizer is not initialized".into()))?;
    state
        .recognizer
        .replace_shards(ShardSet::new(config, shards));
    Ok(())
}

fn validate_shard(weights_path: &str, shard_path: &str) -> Result<()> {
    let decoder = {
        let slot = recognizer()
            .lock()
            .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
        slot.as_ref().map(|state| Arc::clone(&state.decoder))
    };
    let decoder = match decoder {
        Some(decoder) => decoder,
        None => Arc::new(TreeIdDecoder::from_library(weights_path)?),
    };
    let shard_name = Path::new(shard_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| NativeError("shard path has no UTF-8 file name".into()))?;
    Shard::open(shard_name, shard_path, decoder)?.validate()?;
    Ok(())
}

fn recognize(
    samples: &[i16],
    sample_rate: jint,
    run_on_small_cores: jboolean,
    fingerprint_matching_enabled: jboolean,
    previous_match: Option<&PreviousMatch>,
) -> Result<String> {
    let sample_rate = u32::try_from(sample_rate)
        .map_err(|_| NativeError("SoundTrigger supplied a negative sample rate".into()))?;
    let mut slot = recognizer()
        .lock()
        .map_err(|_| NativeError("recognizer state is poisoned".into()))?;
    let recognizer = &mut slot
        .as_mut()
        .ok_or_else(|| NativeError("recognizer is not initialized".into()))?
        .recognizer;
    let _affinity = affinity::Guard::pin_small_cores(run_on_small_cores == JNI_TRUE)?;
    let outcome = recognizer.recognize_pcm_timed_with_context(
        samples,
        sample_rate,
        fingerprint_matching_enabled == JNI_TRUE,
        previous_match,
    )?;
    let timings = outcome.timings;
    *last_timings()
        .lock()
        .map_err(|_| NativeError("recognition timing state is poisoned".into()))? = format!(
        "resampleUs {} musicGateUs {} frontendUs {} embedUs {} searchUs {} scoreUs {} totalUs {}",
        timings.resample.as_micros(),
        timings.music_gate.as_micros(),
        timings.frontend.as_micros(),
        timings.embed.as_micros(),
        timings.search.as_micros(),
        timings.score.as_micros(),
        timings.total.as_micros(),
    );
    let music_score = outcome
        .music_score
        .map_or_else(|| "null".into(), |score| score.to_string());
    let continuity = outcome.continuity.map_or_else(
        || "null".into(),
        |score| {
            format!(
                "{{\"shard\":{},\"numericId\":{},\"score\":{},\"offset\":{}}}",
                json_string(&score.shard_name),
                score.track_id,
                score.score,
                score.offset_seconds,
            )
        },
    );
    let recognition = outcome.recognition.map_or_else(
        || "null".into(),
        |recognition| {
            let result = recognition.result;
            let duration = result
                .metadata
                .duration_seconds
                .map_or_else(|| "null".into(), |duration| duration.to_string());
            format!(
                concat!(
                    "{{\"title\":{},\"artist\":{},\"trackId\":{},",
                    "\"numericId\":{},\"shard\":{},\"score\":{},",
                    "\"offset\":{},\"duration\":{}}}"
                ),
                json_string(&result.metadata.title),
                json_string(&result.metadata.artist),
                json_string(&result.metadata.track_id),
                result.track_id,
                json_string(&result.shard_name),
                result.score,
                result.offset_seconds,
                duration,
            )
        },
    );
    Ok(format!(
        "{{\"musicScore\":{music_score},\"continuity\":{continuity},\"match\":{recognition}}}"
    ))
}

fn json_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write;
                let _ = write!(encoded, "\\u{:04x}", u32::from(character));
            }
            character => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

#[expect(unsafe_code, reason = "JNI export")]
#[allow(missing_docs)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_benzeneos_nowplaying_NativeRecognizer_nativeTimings<'local>(
    mut env: JNIEnv<'local>,
    _receiver: JObject<'local>,
) -> jstring {
    let result = catch_native(|| {
        let timings = last_timings()
            .lock()
            .map_err(|_| NativeError("recognition timing state is poisoned".into()))?;
        env.new_string(&*timings)
            .map(|value| value.into_raw())
            .map_err(Into::into)
    });
    resolve_or_throw(&mut env, result)
}

fn panic_error(payload: Box<dyn Any + Send>) -> NativeError {
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic");
    NativeError(format!("native recognizer panicked, {message}"))
}

fn catch_native<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|payload| Err(panic_error(payload)))
}

fn resolve_or_throw<T: Default>(env: &mut JNIEnv<'_>, result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            if !env.exception_check().unwrap_or(false) {
                let _ = env.throw_new("java/lang/RuntimeException", error.to_string());
            }
            T::default()
        }
    }
}

#[expect(unsafe_code, reason = "JNI export")]
#[unsafe(no_mangle)]
/// Initializes the process-wide recognizer from weight, config, and shard paths.
///
/// Initialization is idempotent. The first successful call installs one recognizer and later
/// calls leave it unchanged. Invalid Java strings, unreadable files, malformed model data, and
/// caught panics are reported as `RuntimeException`.
pub extern "system" fn Java_com_benzeneos_nowplaying_NativeRecognizer_nativeInitialize<'local>(
    mut env: JNIEnv<'local>,
    _receiver: JObject<'local>,
    weights_path: JString<'local>,
    config_path: JString<'local>,
    shard_paths: JObjectArray<'local>,
) {
    let result = catch_native(|| {
        let weights_path = read_string(&mut env, &weights_path)?;
        let config_path = read_string(&mut env, &config_path)?;
        let paths = read_paths(&mut env, &shard_paths)?;
        initialize(&weights_path, &config_path, &paths)
    });
    resolve_or_throw(&mut env, result);
}

#[expect(unsafe_code, reason = "JNI export")]
#[allow(missing_docs)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_benzeneos_nowplaying_NativeRecognizer_nativeReload<'local>(
    mut env: JNIEnv<'local>,
    _receiver: JObject<'local>,
    shard_paths: JObjectArray<'local>,
) {
    let result = catch_native(|| {
        let paths = read_paths(&mut env, &shard_paths)?;
        reload(&paths)
    });
    resolve_or_throw(&mut env, result);
}

#[expect(unsafe_code, reason = "JNI export")]
#[allow(missing_docs)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_benzeneos_nowplaying_NativeRecognizer_nativeValidateShard<
    'local,
>(
    mut env: JNIEnv<'local>,
    _receiver: JObject<'local>,
    weights_path: JString<'local>,
    shard_path: JString<'local>,
) {
    let result = catch_native(|| {
        let weights_path = read_string(&mut env, &weights_path)?;
        let shard_path = read_string(&mut env, &shard_path)?;
        validate_shard(&weights_path, &shard_path)
    });
    resolve_or_throw(&mut env, result);
}

#[expect(unsafe_code, reason = "JNI export")]
#[unsafe(no_mangle)]
/// Recognizes one PCM16 capture at the rate supplied by SoundTrigger.
///
/// Returns JSON containing the music score, continuity diagnostic, and accepted catalog match.
/// Fingerprint matching can be disabled while retaining music scoring. An invalid sample rate,
/// missing initialization, recognition error, or caught panic is reported as `RuntimeException`.
pub extern "system" fn Java_com_benzeneos_nowplaying_NativeRecognizer_nativeRecognize<'local>(
    mut env: JNIEnv<'local>,
    _receiver: JObject<'local>,
    samples: JShortArray<'local>,
    sample_rate: jint,
    run_on_small_cores: jboolean,
    fingerprint_matching_enabled: jboolean,
    previous_shard: JString<'local>,
    previous_track_id: jint,
    previous_offsets: JDoubleArray<'local>,
) -> jstring {
    let result = catch_native(|| {
        let sample_count = usize::try_from(env.get_array_length(&samples)?)
            .map_err(|_| NativeError("PCM sample count does not fit usize".into()))?;
        let mut pcm = vec![0i16; sample_count];
        env.get_short_array_region(&samples, 0, &mut pcm)?;
        let previous = if previous_shard.is_null() {
            None
        } else {
            let shard_name = read_string(&mut env, &previous_shard)?;
            let track_id = u32::try_from(previous_track_id)
                .map_err(|_| NativeError("previous track ID is negative".into()))?;
            let count = usize::try_from(env.get_array_length(&previous_offsets)?)
                .map_err(|_| NativeError("previous offset count does not fit usize".into()))?;
            let mut predicted_offsets_seconds = vec![0.0; count];
            env.get_double_array_region(&previous_offsets, 0, &mut predicted_offsets_seconds)?;
            Some(PreviousMatch {
                shard_name,
                track_id,
                predicted_offsets_seconds,
            })
        };
        let recognition = recognize(
            &pcm,
            sample_rate,
            run_on_small_cores,
            fingerprint_matching_enabled,
            previous.as_ref(),
        )?;
        env.new_string(recognition)
            .map(|value| value.into_raw())
            .map_err(Into::into)
    });
    resolve_or_throw(&mut env, result)
}
