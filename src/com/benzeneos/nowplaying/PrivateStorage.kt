package com.benzeneos.nowplaying

import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.system.Os
import java.io.File

object PrivateStorage {
    private const val OWNER_ONLY = 448

    fun deviceProtectedContext(context: Context): Context {
        val protectedContext = context.createDeviceProtectedStorageContext()
        ensureDirectory(protectedContext.filesDir)
        return protectedContext
    }

    fun ensureDirectory(directory: File): File {
        check(directory.isDirectory || directory.mkdirs()) {
            "Private storage directory is unavailable"
        }
        Os.chmod(directory.path, OWNER_ONLY)
        return directory
    }
}

class NowPlayingApplication : Application() {
    private val userLifecycleReceiver = BootReceiver()

    override fun onCreate() {
        super.onCreate()
        PrivateStorage.deviceProtectedContext(this)
        registerReceiver(
            userLifecycleReceiver,
            IntentFilter().apply {
                addAction(Intent.ACTION_USER_BACKGROUND)
                addAction(Intent.ACTION_USER_FOREGROUND)
            },
            Context.RECEIVER_EXPORTED,
        )
    }
}
