package com.benzeneos.nowplaying

import android.hardware.soundtrigger.SoundTrigger
import android.media.soundtrigger.SoundTriggerDetectionService
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

class NowPlayingDetectionService : SoundTriggerDetectionService() {
    private class Operation {
        val stopped = AtomicBoolean(false)
        var future: Future<*>? = null
        var squelchDecision: SquelchDecision? = null
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val operations = ConcurrentHashMap<Int, Operation>()

    override fun onGenericRecognitionEvent(
        uuid: UUID,
        params: Bundle?,
        opId: Int,
        event: SoundTrigger.RecognitionEvent,
    ) {
        SoundTriggerController.markTriggered()
        val triggerElapsedRealtimeMs = SystemClock.elapsedRealtime()
        val operation = Operation()
        operations[opId] = operation
        operation.future = executor.submit {
            val powerManager = checkNotNull(getSystemService(PowerManager::class.java)) {
                "PowerManager is unavailable"
            }
            val wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "$packageName:recognition",
            )
            wakeLock.acquire(30_000L)
            try {
                operation.squelchDecision = handleRecognition(
                    uuid,
                    opId,
                    event,
                    triggerElapsedRealtimeMs,
                )
            } catch (error: Throwable) {
                Log.e(TAG, "recognition failed", error)
            } finally {
                if (wakeLock.isHeld) {
                    wakeLock.release()
                }
                operations.remove(opId)
                if (!operation.stopped.get()) {
                    SoundTriggerController.armAfter(
                        applicationContext,
                        triggerElapsedRealtimeMs,
                        operation.squelchDecision ?: RecognitionState.currentSquelch(this),
                    )
                    operationFinished(uuid, opId)
                }
            }
        }
    }

    override fun onError(uuid: UUID, params: Bundle?, opId: Int, status: Int) {
        SoundTriggerController.markTriggered()
        val triggerElapsedRealtimeMs = SystemClock.elapsedRealtime()
        Log.e(TAG, "SoundTrigger reported status $status")
        SoundTriggerController.armAfter(
            applicationContext,
            triggerElapsedRealtimeMs,
            RecognitionState.currentSquelch(this),
        )
        operationFinished(uuid, opId)
    }

    override fun onStopOperation(uuid: UUID, params: Bundle?, opId: Int) {
        operations.remove(opId)?.let { operation ->
            operation.stopped.set(true)
            operation.future?.cancel(true)
        }
        Log.w(TAG, "SoundTrigger operation $opId timed out")
        SoundTriggerController.armAfter(
            applicationContext,
            SystemClock.elapsedRealtime(),
            RecognitionState.currentSquelch(this),
        )
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }

    private fun handleRecognition(
        uuid: UUID,
        opId: Int,
        event: SoundTrigger.RecognitionEvent,
        triggerElapsedRealtimeMs: Long,
    ): SquelchDecision? {
        val format = event.captureFormat
        Log.i(
            TAG,
            "SoundTrigger event sampleRate ${format?.sampleRate} encoding ${format?.encoding} " +
                "channels ${format?.channelCount} session ${event.captureSession} " +
                "delayMs ${eventIntField(event, "captureDelayMs")} " +
                "preambleMs ${eventIntField(event, "capturePreambleMs")}",
        )
        RecognitionGates.captureBlockReason(this)?.let { reason ->
            Log.i(TAG, "recognition suppressed because $reason")
            return null
        }
        val playbackPackage = RecognitionGates.activeMusicPlaybackPackage(this)
        val capture = AudioCapture.read(event)
        Log.i(
            TAG,
            "captured ${capture.samples.size} samples at ${capture.sampleRate} Hz",
        )
        val storedCapture = LiveParityCapture.persist(this, uuid, opId, event, capture)
        val fingerprintMatchingEnabled = playbackPackage == null
        val previousMatch = if (fingerprintMatchingEnabled) {
            RecognitionState.projectedPreviousMatch(this)
        } else {
            null
        }
        val result = NativeRecognizer.recognize(
            this,
            capture.samples,
            capture.sampleRate,
            fingerprintMatchingEnabled,
            previousMatch,
        )
        storedCapture?.finish(
            if (playbackPackage == null) result.toString() else {
                "suppressed playback $playbackPackage musicScore ${result.musicScore}"
            },
        )
        if (playbackPackage != null) {
            Log.i(
                TAG,
                "catalog matching suppressed for playback by $playbackPackage " +
                    "music score ${result.musicScore}",
            )
        } else if (result.match == null) {
            Log.i(TAG, "no catalog match")
        } else {
            Log.i(TAG, "catalog match ${result.match}")
        }
        result.continuity?.let { continuity ->
            Log.i(
                TAG,
                "previous match continuity shard ${continuity.shard} " +
                    "track ${continuity.numericId} score ${continuity.score} " +
                    "offset ${continuity.offsetSeconds}",
            )
        }
        val decision = RecognitionState.record(this, result, triggerElapsedRealtimeMs)
        result.match?.let { match ->
            persistHistory(match)
            AmbientIndication.show(this, match, decision.nextRecognitionDelayMs + 10_000L)
        } ?: AmbientIndication.hide(this)
        return decision
    }

    private fun persistHistory(match: CatalogMatch) {
        try {
            RecognitionHistoryStore(this).use { store -> store.record(match) }
        } catch (error: Throwable) {
            Log.e(TAG, "recognition history insert failed", error)
        }
    }

    private fun eventIntField(event: SoundTrigger.RecognitionEvent, name: String): Int? {
        return runCatching { event.javaClass.getField(name).getInt(event) }.getOrNull()
    }

    private companion object {
        const val TAG = "NowPlaying"
    }
}
