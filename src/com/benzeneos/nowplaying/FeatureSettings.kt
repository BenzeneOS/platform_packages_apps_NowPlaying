package com.benzeneos.nowplaying

import android.content.Context
import android.provider.Settings

object FeatureSettings {
    const val ENABLED = "now_playing_enabled"
    const val STORAGE_CAP_MIB = "now_playing_storage_cap_mib"
    private const val BYTES_PER_MIB = 1_048_576L

    fun isEnabled(context: Context): Boolean =
        Settings.Global.getInt(context.contentResolver, ENABLED, 0) != 0

    fun storageCapBytes(context: Context): Long? {
        val capMib = Settings.Global.getLong(context.contentResolver, STORAGE_CAP_MIB, 0L)
        return if (capMib <= 0L) null else Math.multiplyExact(capMib, BYTES_PER_MIB)
    }
}
