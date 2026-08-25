package com.benzeneos.nowplaying

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.os.UserHandle
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

class ShardDownloadJobService : JobService() {
    private class Operation {
        val stopped = AtomicBoolean(false)
        var future: Future<*>? = null
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val running = ConcurrentHashMap<Int, Operation>()

    override fun onStartJob(params: JobParameters): Boolean {
        if (params.jobId == SoundTriggerRearmJob.JOB_ID) {
            return startRearmJob(params)
        }
        if (!SYNC_IN_PROGRESS.compareAndSet(false, true)) {
            return false
        }
        val operation = Operation()
        running[params.jobId] = operation
        operation.future = executor.submit {
            var retry = false
            try {
                if (params.jobId == JOB_ID && !FeatureSettings.isEnabled(this)) {
                    return@submit
                }
                val country = Locale.getDefault().country.lowercase(Locale.ROOT)
                val source = CatalogSource(CatalogSettings.baseUrl(this))
                val result = ShardStore(this).sync(
                    source,
                    country,
                    CatalogSettings.storageCapBytes(this),
                ) { active, downloadedBytes ->
                    Log.i(
                        TAG,
                        "catalog progress build ${active.build} shards ${active.packs.size} " +
                            "downloadedBytes $downloadedBytes",
                    )
                }
                Log.i(
                    TAG,
                    "catalog sync finished build ${result.build} country ${result.country} " +
                        "shards ${result.shardCount} catalogBytes ${result.catalogBytes} " +
                        "downloadedBytes ${result.downloadedBytes} elapsedMs ${result.elapsedMs}",
                )
            } catch (error: Throwable) {
                retry = !operation.stopped.get()
                Log.e(
                    TAG,
                    "catalog sync failed with ${error.javaClass.name}: ${error.message}",
                    error,
                )
            } finally {
                running.remove(params.jobId, operation)
                SYNC_IN_PROGRESS.set(false)
                if (!operation.stopped.get()) {
                    jobFinished(params, retry)
                }
            }
        }
        return true
    }

    private fun startRearmJob(params: JobParameters): Boolean {
        val operation = Operation()
        running[params.jobId] = operation
        Log.i(SOUND_TRIGGER_TAG, "SoundTrigger rearm job started")
        SoundTriggerController.armFromRearmJobAsync(applicationContext) {
            if (running.remove(params.jobId, operation) && !operation.stopped.get()) {
                jobFinished(params, false)
            }
        }
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean {
        running.remove(params.jobId)?.let { operation ->
            operation.stopped.set(true)
            operation.future?.cancel(true)
        }
        return true
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        const val JOB_ID = 0x4e50
        const val REFRESH_JOB_ID = 0x4e51
        private const val TAG = "NowPlayingCatalog"
        private const val SOUND_TRIGGER_TAG = "NowPlaying"
        private const val REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1_000L
        private const val REFRESH_FLEX_MS = 6 * 60 * 60 * 1_000L
        private val SYNC_IN_PROGRESS = AtomicBoolean(false)

        fun schedule(context: Context, immediate: Boolean = false) {
            if (UserHandle.myUserId() != UserHandle.USER_SYSTEM ||
                !FeatureSettings.isEnabled(context)
            ) {
                return
            }
            val scheduler = context.getSystemService(JobScheduler::class.java) ?: return
            if (scheduler.getPendingJob(JOB_ID) == null) {
                val job = JobInfo.Builder(
                    JOB_ID,
                    ComponentName(context, ShardDownloadJobService::class.java),
                )
                    .setRequiredNetworkType(JobInfo.NETWORK_TYPE_UNMETERED)
                    .setPersisted(true)
                    .setPeriodic(REFRESH_INTERVAL_MS, REFRESH_FLEX_MS)
                    .build()
                if (scheduler.schedule(job) != JobScheduler.RESULT_SUCCESS) {
                    Log.e(TAG, "catalog job scheduling failed")
                }
            }
            if (immediate) {
                refresh(context)
            }
        }

        fun reschedule(context: Context) {
            val scheduler = context.getSystemService(JobScheduler::class.java) ?: return
            scheduler.cancel(JOB_ID)
            schedule(context)
        }

        fun refresh(context: Context): Boolean {
            if (UserHandle.myUserId() != UserHandle.USER_SYSTEM) {
                return false
            }
            val scheduler = context.getSystemService(JobScheduler::class.java) ?: return false
            if (scheduler.getPendingJob(REFRESH_JOB_ID) != null) {
                return true
            }
            val job = JobInfo.Builder(
                REFRESH_JOB_ID,
                ComponentName(context, ShardDownloadJobService::class.java),
            )
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_UNMETERED)
                .build()
            return scheduler.schedule(job) == JobScheduler.RESULT_SUCCESS
        }

        fun cancel(context: Context) {
            val scheduler = context.getSystemService(JobScheduler::class.java) ?: return
            scheduler.cancel(JOB_ID)
            scheduler.cancel(REFRESH_JOB_ID)
        }

        fun cancelRefresh(context: Context) {
            context.getSystemService(JobScheduler::class.java)?.cancel(REFRESH_JOB_ID)
        }
    }
}
