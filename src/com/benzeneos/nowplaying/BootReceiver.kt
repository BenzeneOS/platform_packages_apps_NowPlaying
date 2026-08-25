package com.benzeneos.nowplaying

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val enabled = FeatureSettings.isEnabled(context)
        if (enabled) {
            val immediate = shouldRequestImmediateSync(
                intent.action,
                ShardStore(context).isComplete(),
            )
            if (intent.action == Intent.ACTION_LOCALE_CHANGED ||
                intent.action == Intent.ACTION_MY_PACKAGE_REPLACED
            ) {
                ShardDownloadJobService.reschedule(context)
            } else {
                ShardDownloadJobService.schedule(context, immediate)
            }
        } else {
            ShardDownloadJobService.cancel(context)
        }
        val pendingResult = goAsync()
        SoundTriggerController.applyEnabledAsync(context.applicationContext) {
            pendingResult.finish()
        }
    }

    companion object {
        fun shouldRequestImmediateSync(action: String?, catalogComplete: Boolean): Boolean =
            !catalogComplete &&
                (action == Intent.ACTION_LOCKED_BOOT_COMPLETED ||
                    action == Intent.ACTION_BOOT_COMPLETED)
    }
}
