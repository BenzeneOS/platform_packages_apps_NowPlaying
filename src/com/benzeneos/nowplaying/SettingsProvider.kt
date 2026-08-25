package com.benzeneos.nowplaying

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.net.Uri
import android.os.Bundle

class SettingsProvider : ContentProvider() {
    override fun onCreate(): Boolean = true

    override fun call(method: String, arg: String?, extras: Bundle?): Bundle {
        val appContext = requireNotNull(context).applicationContext
        return when (method) {
            METHOD_APPLY_ENABLED -> {
                if (FeatureSettings.isEnabled(appContext)) {
                    ShardDownloadJobService.schedule(appContext, immediate = true)
                } else {
                    ShardDownloadJobService.cancel(appContext)
                }
                Bundle().apply {
                    putBoolean(KEY_SUCCESS, SoundTriggerController.applyEnabledBlocking(appContext))
                }
            }
            METHOD_CATALOG_STATE -> catalogState(appContext)
            METHOD_REFRESH_CATALOG -> Bundle().apply {
                putBoolean(KEY_SUCCESS, ShardDownloadJobService.refresh(appContext))
            }
            METHOD_DELETE_CATALOG -> {
                ShardDownloadJobService.cancelRefresh(appContext)
                val (count, bytes) = ShardStore(appContext).deleteDownloaded()
                catalogState(appContext).apply {
                    putInt(KEY_DELETED_COUNT, count)
                    putLong(KEY_DELETED_BYTES, bytes)
                    putBoolean(KEY_SUCCESS, true)
                }
            }
            else -> throw IllegalArgumentException("Unknown method $method")
        }
    }

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?,
    ): Cursor? = null

    override fun getType(uri: Uri): String? = null

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?,
    ): Int = 0

    private fun catalogState(context: android.content.Context): Bundle {
        val active = ShardStore(context).readActive()
        return Bundle().apply {
            putString(KEY_BUILD, active?.build)
            putString(KEY_COUNTRY, active?.country)
            putInt(KEY_SHARD_COUNT, active?.packs?.size ?: 0)
            putLong(KEY_CATALOG_BYTES, active?.packs?.sumOf(ActivePack::size) ?: 0L)
        }
    }

    companion object {
        const val AUTHORITY = "com.benzeneos.nowplaying.settings"
        const val METHOD_APPLY_ENABLED = "apply_enabled"
        const val METHOD_CATALOG_STATE = "catalog_state"
        const val METHOD_REFRESH_CATALOG = "refresh_catalog"
        const val METHOD_DELETE_CATALOG = "delete_catalog"
        const val KEY_SUCCESS = "success"
        const val KEY_BUILD = "build"
        const val KEY_COUNTRY = "country"
        const val KEY_SHARD_COUNT = "shard_count"
        const val KEY_CATALOG_BYTES = "catalog_bytes"
        const val KEY_DELETED_COUNT = "deleted_count"
        const val KEY_DELETED_BYTES = "deleted_bytes"
    }
}
