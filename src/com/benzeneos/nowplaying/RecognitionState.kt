package com.benzeneos.nowplaying

import android.content.Context
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToLong
import org.json.JSONArray
import org.json.JSONObject

object RecognitionSession {
    private val onDemandSessions = AtomicInteger()

    val onDemandActive: Boolean
        get() = onDemandSessions.get() > 0

    fun beginOnDemand(): AutoCloseable {
        onDemandSessions.incrementAndGet()
        return AutoCloseable {
            onDemandSessions.updateAndGet { count -> (count - 1).coerceAtLeast(0) }
        }
    }
}

data class RecognitionHistoryEntry(
    val elapsedRealtimeMs: Long,
    val matched: Boolean,
    val shard: String?,
    val numericId: Int?,
    val offsetsMs: List<Long>,
    val durationMs: Long?,
)

data class SquelchDecision(
    val squelchMs: Long,
    val nextRecognitionDelayMs: Long,
)

object SquelchPolicy {
    const val POSITIVE_PRE_EVENT_MS = 4_000L
    const val FALLBACK_MS = 55_000L
    private const val HISTORY_EXPIRATION_MS = 1_200_000L
    private const val ADAPTATION_WINDOW_MS = 720_000L
    private const val REPEATED_MISS_COUNT = 10
    private const val REPEATED_MISS_MS = 90_000L
    private const val HEURISTIC_PADDING_MS = 5_000L
    private const val HEURISTIC_THRESHOLD_MS = 50_000L
    private const val RECOGNITION_INPUT_MS = 8_000L
    private const val CANDIDATE_MIN_MS = 20_000L
    private const val CANDIDATE_MAX_MS = 90_000L
    private const val MIN_MS = 15_000L
    private const val MAX_MS = 180_000L

    fun decide(history: List<RecognitionHistoryEntry>, nowMs: Long): SquelchDecision {
        val retained = history.filter { entry ->
            nowMs - entry.elapsedRealtimeMs in 0..HISTORY_EXPIRATION_MS
        }
        val newest = retained.maxByOrNull(RecognitionHistoryEntry::elapsedRealtimeMs)
        val recent = if (newest == null) {
            emptyList()
        } else {
            retained.filter { entry ->
                newest.elapsedRealtimeMs - entry.elapsedRealtimeMs in 0..ADAPTATION_WINDOW_MS
            }
        }
        val candidate = when {
            recent.none(RecognitionHistoryEntry::matched) -> {
                if (recent.size >= REPEATED_MISS_COUNT) REPEATED_MISS_MS else FALLBACK_MS
            }
            newest?.matched != true -> FALLBACK_MS
            else -> matchedSquelch(newest, nowMs)
        }.coerceIn(MIN_MS, MAX_MS)
        return SquelchDecision(candidate, candidate + POSITIVE_PRE_EVENT_MS)
    }

    private fun matchedSquelch(entry: RecognitionHistoryEntry, nowMs: Long): Long {
        val durationMs = entry.durationMs ?: return FALLBACK_MS
        val offsetMs = entry.offsetsMs.maxOrNull() ?: return FALLBACK_MS
        val roundedOffsetMs = (offsetMs.toDouble() / 2_000.0).roundToLong() * 2_000L
        val elapsedSinceMatchMs = (nowMs - entry.elapsedRealtimeMs).coerceAtLeast(0)
        val remainingMs = durationMs - RECOGNITION_INPUT_MS - roundedOffsetMs - elapsedSinceMatchMs
        return if (remainingMs < HEURISTIC_THRESHOLD_MS) {
            (remainingMs + HEURISTIC_PADDING_MS).coerceIn(CANDIDATE_MIN_MS, CANDIDATE_MAX_MS)
        } else {
            FALLBACK_MS
        }
    }
}

object RecognitionState {
    private const val PREFERENCES = "recognition_state"
    private const val HISTORY = "history"
    private const val PREVIOUS_MATCH = "previous_match"
    private const val HISTORY_EXPIRATION_MS = 1_200_000L

    fun projectedPreviousMatch(context: Context): ProjectedPreviousMatch? {
        val nowMs = SystemClock.elapsedRealtime()
        val encoded = preferences(context).getString(PREVIOUS_MATCH, null) ?: return null
        val value = runCatching { JSONObject(encoded) }.getOrNull() ?: return null
        val recordedMs = value.optLong("elapsedMs", -1L)
        if (recordedMs !in 0..nowMs) {
            preferences(context).edit().remove(PREVIOUS_MATCH).apply()
            return null
        }
        val deltaMs = nowMs - recordedMs
        if (deltaMs > HISTORY_EXPIRATION_MS) {
            preferences(context).edit().remove(PREVIOUS_MATCH).apply()
            return null
        }
        val offsets = value.getJSONArray("offsetsMs")
        val projected = DoubleArray(offsets.length()) { index ->
            (offsets.getLong(index) + deltaMs).toDouble() / 1_000.0
        }
        return ProjectedPreviousMatch(
            shard = value.getString("shard"),
            numericId = value.getInt("numericId"),
            offsetsSeconds = projected,
        )
    }

    fun record(
        context: Context,
        recognition: NativeRecognition,
        elapsedRealtimeMs: Long,
    ): SquelchDecision {
        val match = recognition.match
        val entry = RecognitionHistoryEntry(
            elapsedRealtimeMs = elapsedRealtimeMs,
            matched = match != null,
            shard = match?.shard,
            numericId = match?.numericId,
            offsetsMs = match?.let { listOf((it.offsetSeconds * 1_000.0).roundToLong()) }
                ?: emptyList(),
            durationMs = match?.durationSeconds?.let { (it * 1_000.0).roundToLong() },
        )
        val nowMs = SystemClock.elapsedRealtime()
        val history = (readHistory(context) + entry)
            .filter { item -> nowMs - item.elapsedRealtimeMs in 0..HISTORY_EXPIRATION_MS }
            .sortedBy(RecognitionHistoryEntry::elapsedRealtimeMs)
        val editor = preferences(context).edit().putString(HISTORY, encodeHistory(history).toString())
        if (match != null) {
            editor.putString(
                PREVIOUS_MATCH,
                JSONObject()
                    .put("elapsedMs", elapsedRealtimeMs)
                    .put("shard", match.shard)
                    .put("numericId", match.numericId)
                    .put("offsetsMs", JSONArray(entry.offsetsMs))
                    .put("durationMs", entry.durationMs)
                    .toString(),
            )
        }
        editor.apply()
        return SquelchPolicy.decide(history, nowMs)
    }

    fun currentSquelch(context: Context): SquelchDecision =
        SquelchPolicy.decide(readHistory(context), SystemClock.elapsedRealtime())

    private fun readHistory(context: Context): List<RecognitionHistoryEntry> {
        val encoded = preferences(context).getString(HISTORY, null) ?: return emptyList()
        val values = runCatching { JSONArray(encoded) }.getOrNull() ?: return emptyList()
        return buildList {
            for (index in 0 until values.length()) {
                val value = values.getJSONObject(index)
                val offsets = value.getJSONArray("offsetsMs")
                add(
                    RecognitionHistoryEntry(
                        elapsedRealtimeMs = value.getLong("elapsedMs"),
                        matched = value.getBoolean("matched"),
                        shard = value.optString("shard").ifEmpty { null },
                        numericId = if (value.isNull("numericId")) null else value.getInt("numericId"),
                        offsetsMs = buildList {
                            for (offsetIndex in 0 until offsets.length()) {
                                add(offsets.getLong(offsetIndex))
                            }
                        },
                        durationMs = if (value.isNull("durationMs")) {
                            null
                        } else {
                            value.getLong("durationMs")
                        },
                    ),
                )
            }
        }
    }

    private fun encodeHistory(history: List<RecognitionHistoryEntry>): JSONArray =
        JSONArray().apply {
            history.forEach { entry ->
                put(
                    JSONObject()
                        .put("elapsedMs", entry.elapsedRealtimeMs)
                        .put("matched", entry.matched)
                        .put("shard", entry.shard)
                        .put("numericId", entry.numericId)
                        .put("offsetsMs", JSONArray(entry.offsetsMs))
                        .put("durationMs", entry.durationMs),
                )
            }
        }

    private fun preferences(context: Context) = context
        .createDeviceProtectedStorageContext()
        .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
}
