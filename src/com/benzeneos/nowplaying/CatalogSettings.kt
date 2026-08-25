package com.benzeneos.nowplaying

import android.content.Context
import android.net.Uri

object CatalogSettings {
    const val DEFAULT_BASE_URL = "https://storage.googleapis.com"
    private const val PREFERENCES = "catalog_settings"
    private const val BASE_URL = "base_url"

    fun storageCapBytes(context: Context): Long? = FeatureSettings.storageCapBytes(context)

    fun baseUrl(context: Context): String =
        preferences(context).getString(BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL

    internal fun normalizeBaseUrl(value: String): String {
        val normalized = value.trim().trimEnd('/')
        val uri = Uri.parse(normalized)
        require(
            uri.scheme == "https" &&
                !uri.host.isNullOrEmpty() &&
                uri.query == null &&
                uri.fragment == null,
        ) { "Catalog service must be an HTTPS base URL" }
        return normalized
    }

    private fun preferences(context: Context) = context
        .createDeviceProtectedStorageContext()
        .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
}
