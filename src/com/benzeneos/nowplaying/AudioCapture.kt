package com.benzeneos.nowplaying

import android.hardware.soundtrigger.SoundTrigger
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord

data class CapturedAudio(val samples: ShortArray, val sampleRate: Int)

object AudioCapture {
    private const val BUFFER_SECONDS = 8
    private const val HOTWORD_CAPTURE_PRESET = 1999

    fun read(event: SoundTrigger.RecognitionEvent): CapturedAudio {
        check(event.isCaptureAvailable) { "recognition event has no capture session" }
        val format = checkNotNull(event.captureFormat) {
            "recognition event has no capture format"
        }
        check(format.encoding == AudioFormat.ENCODING_PCM_16BIT) {
            "capture encoding ${format.encoding} is not PCM16"
        }
        check(format.channelCount == 1) {
            "capture channel count ${format.channelCount} is not mono"
        }
        val attributesBuilder = AudioAttributes.Builder()
        AudioAttributes.Builder::class.java
            .getMethod("setInternalCapturePreset", Integer.TYPE)
            .invoke(attributesBuilder, HOTWORD_CAPTURE_PRESET)
        val attributes = attributesBuilder.build()
        val bufferBytes = Math.round(
            2.0 * format.sampleRate.toDouble() * BUFFER_SECONDS.toDouble()
        ).toInt()
        val audioRecord = AudioRecord::class.java
            .getConstructor(
                AudioAttributes::class.java,
                AudioFormat::class.java,
                Integer.TYPE,
                Integer.TYPE,
            )
            .newInstance(attributes, format, bufferBytes, event.captureSession) as AudioRecord
        try {
            check(audioRecord.state == AudioRecord.STATE_INITIALIZED) {
                "AudioRecord failed to initialize"
            }
            audioRecord.startRecording()
            val samples = ShortArray(audioRecord.bufferSizeInFrames)
            val samplesRead = audioRecord.read(
                samples,
                0,
                samples.size,
                AudioRecord.READ_BLOCKING,
            )
            check(samplesRead == samples.size) {
                "AudioRecord returned $samplesRead of ${samples.size} samples"
            }
            return CapturedAudio(samples, audioRecord.sampleRate)
        } finally {
            if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                audioRecord.stop()
            }
            audioRecord.release()
        }
    }
}
