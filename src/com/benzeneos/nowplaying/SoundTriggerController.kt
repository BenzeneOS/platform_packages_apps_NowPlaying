package com.benzeneos.nowplaying

import android.content.ComponentName
import android.content.Context
import android.media.soundtrigger.SoundTriggerManager
import android.os.SystemClock
import android.os.UserManager
import android.util.Log
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

object SoundTriggerController {
    private const val MODEL_PATH = "/product/etc/firmware/music_detector.sound_model_tflite"
    private const val TAG = "NowPlaying"
    private val MODEL_UUID = UUID.fromString("9f6ad62a-1f0b-11e7-87c5-40a8f03d3f15")
    private val executor = Executors.newSingleThreadExecutor()
    private var armed = false
    private var loaded = false

    fun initializeAsync(context: Context, completion: (() -> Unit)? = null) {
        executor.execute {
            try {
                NativeRecognizer.initialize(context.applicationContext)
            } catch (error: Throwable) {
                Log.e(TAG, "recognizer initialization failed", error)
            } finally {
                completion?.invoke()
            }
        }
    }

    fun armAsync(context: Context, completion: (() -> Unit)? = null) {
        armAsync(context, true, completion)
    }

    fun armFromRearmJobAsync(context: Context, completion: (() -> Unit)? = null) {
        armAsync(context, false, completion)
    }

    private fun armAsync(
        context: Context,
        cancelPendingRearm: Boolean,
        completion: (() -> Unit)?,
    ) {
        executor.execute {
            try {
                arm(context, cancelPendingRearm)
            } catch (error: Throwable) {
                Log.e(TAG, "SoundTrigger arm failed", error)
            } finally {
                completion?.invoke()
            }
        }
    }

    fun applyEnabledAsync(context: Context, completion: (() -> Unit)? = null) {
        executor.execute {
            try {
                applyEnabled(context.applicationContext)
            } catch (error: Throwable) {
                Log.e(TAG, "SoundTrigger setting application failed", error)
            } finally {
                completion?.invoke()
            }
        }
    }

    fun applyEnabledBlocking(context: Context): Boolean = executor.submit<Boolean> {
        try {
            applyEnabled(context.applicationContext)
            true
        } catch (error: Throwable) {
            Log.e(TAG, "SoundTrigger setting application failed", error)
            false
        }
    }.get()

    @Synchronized
    fun armAfter(
        context: Context,
        triggerElapsedRealtimeMs: Long,
        decision: SquelchDecision,
    ) {
        if (!canArm(context)) {
            return
        }
        val elapsedMs = SystemClock.elapsedRealtime() - triggerElapsedRealtimeMs
        val remainingMs = (decision.squelchMs - elapsedMs).coerceAtLeast(0)
        Log.i(
            TAG,
            "SoundTrigger squelch ${decision.squelchMs} ms next recognition delay " +
                "${decision.nextRecognitionDelayMs} ms rearm in $remainingMs ms",
        )
        SoundTriggerRearmJob.schedule(context, remainingMs)
    }

    @Synchronized
    fun markTriggered() {
        armed = false
    }

    @Synchronized
    private fun arm(context: Context, cancelPendingRearm: Boolean = true) {
        if (!canArm(context)) {
            disarm(context, cancelPendingRearm)
            return
        }
        if (cancelPendingRearm) {
            SoundTriggerRearmJob.cancel(context)
        }
        val manager = context.getSystemService(SoundTriggerManager::class.java)
            ?: error("SoundTriggerManager is unavailable")
        if (armed) {
            Log.i(TAG, "SoundTrigger recognition is already active")
            return
        }
        if (!loaded) {
            loadModel(manager)
            loaded = true
        }
        var status = startRecognition(context, manager)
        if (status != 0) {
            loaded = false
            loadModel(manager)
            loaded = true
            status = startRecognition(context, manager)
        }
        check(status == 0) { "startRecognition failed with status $status" }
        armed = true
        Log.i(TAG, "SoundTrigger recognition armed")
    }

    @Synchronized
    private fun applyEnabled(context: Context) {
        if (FeatureSettings.isEnabled(context)) {
            arm(context)
        } else {
            disarm(context)
        }
    }

    @Synchronized
    private fun disarm(context: Context, cancelPendingRearm: Boolean = true) {
        if (cancelPendingRearm) {
            SoundTriggerRearmJob.cancel(context)
        }
        if (!loaded) {
            armed = false
            return
        }
        val manager = context.getSystemService(SoundTriggerManager::class.java)
            ?: error("SoundTriggerManager is unavailable")
        try {
            if (armed) {
                val stopStatus = manager.stopRecognition(MODEL_UUID)
                if (stopStatus != 0) {
                    Log.w(TAG, "stopRecognition failed with status $stopStatus")
                }
            }
            val status = manager.unloadSoundModel(MODEL_UUID)
            check(status == 0) { "unloadSoundModel failed with status $status" }
        } finally {
            armed = false
            loaded = false
        }
        Log.i(TAG, "SoundTrigger recognition disabled")
    }

    private fun canArm(context: Context): Boolean {
        val userManager = context.getSystemService(UserManager::class.java)
        return canArm(
            userManager?.isSystemUser == true,
            userManager?.isUserForeground == true,
            FeatureSettings.isEnabled(context),
        )
    }

    fun canArm(systemUser: Boolean, foreground: Boolean, enabled: Boolean): Boolean =
        systemUser && foreground && enabled

    private fun loadModel(manager: SoundTriggerManager) {
        val data = File(MODEL_PATH).readBytes()
        invokeLoadModel(manager, null, byteArrayOf())
        val status = invokeLoadModel(manager, MODEL_UUID, data)
        check(status == 0) { "loadSoundModel failed with status $status" }
        Log.i(TAG, "SoundTrigger model loaded from $MODEL_PATH")
    }

    private fun invokeLoadModel(
        manager: SoundTriggerManager,
        vendorUuid: UUID?,
        data: ByteArray,
    ): Int {
        val soundModelClass = Class.forName(
            "android.hardware.soundtrigger.SoundTrigger\$SoundModel"
        )
        val genericModelClass = Class.forName(
            "android.hardware.soundtrigger.SoundTrigger\$GenericSoundModel"
        )
        val model = genericModelClass
            .getConstructor(UUID::class.java, UUID::class.java, ByteArray::class.java)
            .newInstance(MODEL_UUID, vendorUuid, data)
        return (SoundTriggerManager::class.java
            .getMethod("loadSoundModel", soundModelClass)
            .invoke(manager, model) as Number).toInt()
    }

    private fun startRecognition(context: Context, manager: SoundTriggerManager): Int {
        val keyphraseExtrasClass = Class.forName(
            "[Landroid.hardware.soundtrigger.SoundTrigger\$KeyphraseRecognitionExtra;"
        )
        val recognitionConfigClass = Class.forName(
            "android.hardware.soundtrigger.SoundTrigger\$RecognitionConfig"
        )
        val config = recognitionConfigClass
            .getConstructor(
                java.lang.Boolean.TYPE,
                java.lang.Boolean.TYPE,
                keyphraseExtrasClass,
                ByteArray::class.java,
            )
            .newInstance(true, false, null, null)
        val component = ComponentName(context, NowPlayingDetectionService::class.java)
        return (SoundTriggerManager::class.java
            .getMethod(
                "startRecognition",
                UUID::class.java,
                android.os.Bundle::class.java,
                ComponentName::class.java,
                recognitionConfigClass,
            )
            .invoke(manager, MODEL_UUID, null, component, config) as Number).toInt()
    }
}
