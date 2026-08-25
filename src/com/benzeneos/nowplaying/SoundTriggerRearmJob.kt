package com.benzeneos.nowplaying

import android.app.job.JobInfo
import android.app.job.JobScheduler
import android.content.ComponentName
import android.content.Context
import android.util.Log

object SoundTriggerRearmJob {
    const val JOB_ID = 0x4e52
    private const val TAG = "NowPlaying"

    fun schedule(context: Context, delayMs: Long): Boolean {
        val scheduler = context.getSystemService(JobScheduler::class.java) ?: return false
        val scheduled = try {
            scheduler.schedule(jobInfo(context, delayMs)) == JobScheduler.RESULT_SUCCESS
        } catch (error: RuntimeException) {
            Log.e(TAG, "SoundTrigger rearm job scheduling failed", error)
            return false
        }
        if (!scheduled) {
            Log.e(TAG, "SoundTrigger rearm job scheduling failed")
        }
        return scheduled
    }

    fun cancel(context: Context) {
        context.getSystemService(JobScheduler::class.java)?.cancel(JOB_ID)
    }

    fun jobInfo(context: Context, delayMs: Long): JobInfo {
        val boundedDelayMs = delayMs.coerceAtLeast(0)
        return JobInfo.Builder(
            JOB_ID,
            ComponentName(context, ShardDownloadJobService::class.java),
        )
            .setMinimumLatency(boundedDelayMs)
            .setOverrideDeadline(boundedDelayMs)
            .build()
    }
}
