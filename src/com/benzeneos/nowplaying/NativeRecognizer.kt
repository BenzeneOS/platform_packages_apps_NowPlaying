package com.benzeneos.nowplaying

import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.File
import org.json.JSONObject

data class ProjectedPreviousMatch(
    val shard: String,
    val numericId: Int,
    val offsetsSeconds: DoubleArray,
)

data class CatalogMatch(
    val title: String,
    val artist: String,
    val trackId: String,
    val numericId: Int,
    val shard: String,
    val score: Double,
    val offsetSeconds: Double,
    val durationSeconds: Double?,
) {
    override fun toString(): String =
        "$title by $artist | track $trackId | shard $shard | score $score | offset $offsetSeconds"
}

data class ContinuityDiagnostic(
    val shard: String,
    val numericId: Int,
    val score: Double,
    val offsetSeconds: Double,
)

data class NativeRecognition(
    val match: CatalogMatch?,
    val musicScore: Double?,
    val continuity: ContinuityDiagnostic?,
) {
    override fun toString(): String = match?.toString() ?: "no catalog match"
}

object NativeRecognizer {
    private const val CONFIG_PATH = "/system/etc/nowplaying/v3_config_tah.pb"
    private const val CORE_SHARD_PATH = "/product/etc/ambient/matcher_tah.leveldb"
    private const val WEIGHTS_PATH = "/system/etc/nowplaying/nnfp_v3.weights"

    private var initialized = false

    init {
        System.loadLibrary("nowplaying_jni")
    }

    @Synchronized
    fun initialize(context: Context) {
        if (!initialized) {
            val started = SystemClock.elapsedRealtimeNanos()
            val shardPaths = shardPaths(ShardStore(context).activeFiles())
            try {
                nativeInitialize(WEIGHTS_PATH, CONFIG_PATH, shardPaths)
            } catch (error: RuntimeException) {
                Log.e(TAG, "downloaded shard set was rejected during initialization", error)
                nativeInitialize(WEIGHTS_PATH, CONFIG_PATH, arrayOf(CORE_SHARD_PATH))
            }
            initialized = true
            Log.i(TAG, "recognition timing setupUs ${(SystemClock.elapsedRealtimeNanos() - started) / 1_000}")
        }
    }

    @Synchronized
    fun recognize(
        context: Context,
        samples: ShortArray,
        sampleRate: Int,
        fingerprintMatchingEnabled: Boolean,
        previousMatch: ProjectedPreviousMatch?,
    ): NativeRecognition {
        initialize(context)
        val response = nativeRecognize(
            samples,
            sampleRate,
            RUN_ON_SMALL_CORES,
            fingerprintMatchingEnabled,
            previousMatch?.shard,
            previousMatch?.numericId ?: -1,
            previousMatch?.offsetsSeconds ?: doubleArrayOf(),
        )
        Log.i(TAG, "recognition timing ${nativeTimings()}")
        return parseResponse(response)
    }

    @Synchronized
    fun replaceShards(files: List<File>) {
        val paths = shardPaths(files)
        if (initialized) {
            nativeReload(paths)
        } else {
            nativeInitialize(WEIGHTS_PATH, CONFIG_PATH, paths)
            initialized = true
        }
    }

    @Synchronized
    fun dropDownloadedShards() {
        if (initialized) {
            nativeReload(arrayOf(CORE_SHARD_PATH))
        }
    }

    fun validateShard(file: File) {
        nativeValidateShard(WEIGHTS_PATH, file.path)
    }

    private external fun nativeInitialize(
        weightsPath: String,
        configPath: String,
        shardPaths: Array<String>,
    )

    private external fun nativeReload(shardPaths: Array<String>)

    private external fun nativeValidateShard(weightsPath: String, shardPath: String)

    private external fun nativeRecognize(
        samples: ShortArray,
        sampleRate: Int,
        runOnSmallCores: Boolean,
        fingerprintMatchingEnabled: Boolean,
        previousShard: String?,
        previousNumericId: Int,
        previousOffsetsSeconds: DoubleArray,
    ): String

    private external fun nativeTimings(): String

    private fun shardPaths(files: List<File>): Array<String> =
        (listOf(CORE_SHARD_PATH) + files.map(File::getPath)).toTypedArray()

    private fun parseResponse(response: String): NativeRecognition {
        val root = JSONObject(response)
        val match = root.optJSONObject("match")?.let { value ->
            CatalogMatch(
                title = value.getString("title"),
                artist = value.getString("artist"),
                trackId = value.getString("trackId"),
                numericId = value.getInt("numericId"),
                shard = value.getString("shard"),
                score = value.getDouble("score"),
                offsetSeconds = value.getDouble("offset"),
                durationSeconds = value.optionalDouble("duration"),
            )
        }
        val continuity = root.optJSONObject("continuity")?.let { value ->
            ContinuityDiagnostic(
                shard = value.getString("shard"),
                numericId = value.getInt("numericId"),
                score = value.getDouble("score"),
                offsetSeconds = value.getDouble("offset"),
            )
        }
        return NativeRecognition(
            match = match,
            musicScore = root.optionalDouble("musicScore"),
            continuity = continuity,
        )
    }

    private fun JSONObject.optionalDouble(name: String): Double? =
        if (isNull(name)) null else getDouble(name)

    private const val TAG = "NowPlaying"
    private const val RUN_ON_SMALL_CORES = false
}
