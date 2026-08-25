package com.benzeneos.nowplaying

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.AudioPlaybackConfiguration
import android.os.UserHandle
import android.os.UserManager
import android.telephony.TelephonyManager

object RecognitionGates {
    private val playbackIgnorelist = listOf(
        "ak.alizandro.smartaudiobookplayer",
        "com.breel.wallpapers",
        "com.google.android.googlequicksearchbox",
    )

    fun captureBlockReason(context: Context): String? {
        if (RecognitionSession.onDemandActive) {
            return "an on-demand session is active"
        }
        val userManager = context.getSystemService(UserManager::class.java)
        if (userManager?.isUserForeground != true) {
            return "the app user is not foreground"
        }
        val mainUser = userManager.mainUser
        if (mainUser != null && mainUser.identifier != UserHandle.myUserId()) {
            val mainUserManager = context
                .createContextAsUser(mainUser, 0)
                .getSystemService(UserManager::class.java)
            if (mainUserManager?.isUserForeground != true) {
                return "the main user is not foreground"
            }
        }
        if (context.checkSelfPermission(Manifest.permission.READ_PHONE_STATE) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return "phone state is unavailable"
        }
        val telephonyManager = context.getSystemService(TelephonyManager::class.java)
        val callState = runCatching { telephonyManager?.callState }.getOrNull()
        if (callState == null) {
            return "phone state is unavailable"
        }
        if (callState != TelephonyManager.CALL_STATE_IDLE) {
            return "a phone call is active"
        }
        val audioManager = context.getSystemService(AudioManager::class.java)
            ?: return "AudioManager is unavailable"
        if (audioManager.activeRecordingConfigurations.isNotEmpty()) {
            return "another audio recording is active"
        }
        return null
    }

    fun activeMusicPlaybackPackage(context: Context): String? {
        val audioManager = context.getSystemService(AudioManager::class.java) ?: return null
        return audioManager.activePlaybackConfigurations
            .firstOrNull { configuration -> isMusicPlayback(context, configuration) }
            ?.let { configuration ->
                context.packageManager.getNameForUid(configuration.clientUid)
                    ?: configuration.clientUid.toString()
            }
    }

    private fun isMusicPlayback(
        context: Context,
        configuration: AudioPlaybackConfiguration,
    ): Boolean {
        val packageName = context.packageManager.getNameForUid(configuration.clientUid)
        val attributes = configuration.audioAttributes
        return isMusicPlayback(
            packageName,
            attributes.usage,
            attributes.contentType,
            configuration.playerState,
        )
    }

    fun isMusicPlayback(
        packageName: String?,
        usage: Int,
        contentType: Int,
        playerState: Int,
    ): Boolean {
        if (packageName != null && playbackIgnorelist.any(packageName::startsWith)) {
            return false
        }
        return usage == AudioAttributes.USAGE_MEDIA &&
            contentType == AudioAttributes.CONTENT_TYPE_MUSIC &&
            playerState == AudioPlaybackConfiguration.PLAYER_STATE_STARTED
    }
}
