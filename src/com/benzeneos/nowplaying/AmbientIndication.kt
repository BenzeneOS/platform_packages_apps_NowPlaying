package com.benzeneos.nowplaying

import android.app.PendingIntent
import android.content.Context
import android.content.Intent

object AmbientIndication {
    private const val ACTION_HIDE =
        "com.google.android.ambientindication.action.AMBIENT_INDICATION_HIDE"
    private const val ACTION_SHOW =
        "com.google.android.ambientindication.action.AMBIENT_INDICATION_SHOW"
    private const val EXTRA_OPEN_INTENT =
        "com.google.android.ambientindication.extra.OPEN_INTENT"
    private const val EXTRA_TEXT = "com.google.android.ambientindication.extra.TEXT"
    private const val EXTRA_TTL_MILLIS =
        "com.google.android.ambientindication.extra.TTL_MILLIS"
    private const val EXTRA_VERSION = "com.google.android.ambientindication.extra.VERSION"
    private const val SYSTEM_UI_PACKAGE = "com.android.systemui"
    private const val VERSION = 1

    fun show(context: Context, match: CatalogMatch, ttlMillis: Long) {
        val openHistory = PendingIntent.getActivity(
            context,
            0,
            Intent(context, HistoryActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val text = if (match.artist.isBlank()) match.title else "${match.title} • ${match.artist}"
        context.sendBroadcast(
            Intent(ACTION_SHOW)
                .setPackage(SYSTEM_UI_PACKAGE)
                .putExtra(EXTRA_VERSION, VERSION)
                .putExtra(EXTRA_TEXT, text)
                .putExtra(EXTRA_TTL_MILLIS, ttlMillis)
                .putExtra(EXTRA_OPEN_INTENT, openHistory)
                .addFlags(Intent.FLAG_RECEIVER_FOREGROUND),
        )
    }

    fun hide(context: Context) {
        context.sendBroadcast(
            Intent(ACTION_HIDE)
                .setPackage(SYSTEM_UI_PACKAGE)
                .putExtra(EXTRA_VERSION, VERSION)
                .addFlags(Intent.FLAG_RECEIVER_FOREGROUND),
        )
    }
}
