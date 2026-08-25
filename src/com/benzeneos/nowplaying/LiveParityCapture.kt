package com.benzeneos.nowplaying

import android.content.Context
import android.hardware.soundtrigger.SoundTrigger
import android.os.SystemClock
import android.os.SystemProperties
import android.os.UserHandle
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.UUID

object LiveParityCapture {
    private const val PROPERTY = "persist.sys.nowplaying.live_parity"
    private const val TAG = "NowPlaying"

    class StoredCapture(private val metadata: File) {
        fun finish(result: String?) {
            metadata.appendText(
                "rustResult=${result?.replace('\n', ' ') ?: "none"}\n",
                StandardCharsets.UTF_8,
            )
        }
    }

    fun persist(
        context: Context,
        uuid: UUID,
        opId: Int,
        event: SoundTrigger.RecognitionEvent,
        capture: CapturedAudio,
    ): StoredCapture? {
        if (!SystemProperties.getBoolean(PROPERTY, false)) {
            return null
        }
        val directory = File(context.filesDir, "live-parity")
        check(directory.exists() || directory.mkdirs()) {
            "cannot create live parity capture directory"
        }
        val name = "capture-${System.currentTimeMillis()}-$opId"
        val wav = File(directory, "$name.wav")
        val metadata = File(directory, "$name.meta")
        wav.writeBytes(wavBytes(capture))
        metadata.writeText(
            buildString {
                appendLine("capturedAtUnixMs=${System.currentTimeMillis()}")
                appendLine("elapsedRealtimeNanos=${SystemClock.elapsedRealtimeNanos()}")
                appendLine("userId=${UserHandle.myUserId()}")
                appendLine("operationId=$opId")
                appendLine("modelUuid=$uuid")
                appendLine("eventStatus=${eventIntField(event, "status")}")
                appendLine("eventType=${eventIntField(event, "type")}")
                appendLine("soundModelHandle=${eventIntField(event, "soundModelHandle")}")
                appendLine("captureSession=${event.captureSession}")
                appendLine("captureDelayMs=${eventIntField(event, "captureDelayMs")}")
                appendLine("capturePreambleMs=${eventIntField(event, "capturePreambleMs")}")
                appendLine("sampleRate=${capture.sampleRate}")
                appendLine("encoding=${event.captureFormat?.encoding}")
                appendLine("channelCount=${event.captureFormat?.channelCount}")
                appendLine("sampleCount=${capture.samples.size}")
                appendLine("eventData=${eventBytes(event).toHex()}")
            },
            StandardCharsets.UTF_8,
        )
        Log.i(TAG, "live parity capture ${wav.path}")
        return StoredCapture(metadata)
    }

    private fun wavBytes(capture: CapturedAudio): ByteArray {
        val dataSize = capture.samples.size * Short.SIZE_BYTES
        val bytes = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        bytes.put("RIFF".toByteArray(StandardCharsets.US_ASCII))
        bytes.putInt(36 + dataSize)
        bytes.put("WAVEfmt ".toByteArray(StandardCharsets.US_ASCII))
        bytes.putInt(16)
        bytes.putShort(1)
        bytes.putShort(1)
        bytes.putInt(capture.sampleRate)
        bytes.putInt(capture.sampleRate * Short.SIZE_BYTES)
        bytes.putShort(Short.SIZE_BYTES.toShort())
        bytes.putShort(Short.SIZE_BITS.toShort())
        bytes.put("data".toByteArray(StandardCharsets.US_ASCII))
        bytes.putInt(dataSize)
        capture.samples.forEach { bytes.putShort(it) }
        return bytes.array()
    }

    private fun eventIntField(event: SoundTrigger.RecognitionEvent, name: String): Int? {
        return runCatching { event.javaClass.getField(name).getInt(event) }.getOrNull()
    }

    private fun eventBytes(event: SoundTrigger.RecognitionEvent): ByteArray {
        return runCatching {
            event.javaClass.getField("data").get(event) as ByteArray
        }.getOrDefault(byteArrayOf())
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
