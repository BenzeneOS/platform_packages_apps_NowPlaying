package com.benzeneos.nowplaying

import android.Manifest
import android.app.job.JobScheduler
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioPlaybackConfiguration
import android.system.Os
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RecoveredBehaviorTest {
    @Test
    fun catalogManifestKeepsLocaleOrderAndStopsAtTheCap() {
        val manifest = CatalogManifestParser.parse(
            "20260823-030028",
            """{
                "packs": [
                    {
                        "name": "US00000000000000000000000000000000",
                        "size": 5,
                        "compressed_size": 5,
                        "download_urls": ["https://example.com/one"],
                        "extra_type": "shard",
                        "extra_country_codes": "us,xa"
                    },
                    {
                        "name": "CA00000000000000000000000000000000",
                        "size": 3,
                        "compressed_size": 3,
                        "download_urls": ["https://example.com/two"],
                        "extra_type": "shard",
                        "extra_country_codes": "ca"
                    },
                    {
                        "name": "US11111111111111111111111111111111",
                        "size": 7,
                        "compressed_size": 7,
                        "download_urls": ["https://example.com/three"],
                        "extra_type": "shard",
                        "extra_country_codes": "us"
                    }
                ]
            }""".trimIndent(),
        )

        assertEquals(
            listOf("US00000000000000000000000000000000"),
            manifest.select("us", 10L).map(CatalogPack::name),
        )
        assertEquals(2, manifest.select("us", null).size)
    }

    @Test
    fun catalogManifestRejectsMalformedNetworkFields() {
        val malformed = """{
            "packs": [{
                "name": "../shard",
                "size": 4,
                "compressed_size": 4,
                "download_urls": ["http://example.com/shard"],
                "extra_type": "shard",
                "extra_country_codes": "us"
            }]
        }""".trimIndent()

        var rejected = false
        try {
            CatalogManifestParser.parse("20260823-030028", malformed)
        } catch (_: IllegalArgumentException) {
            rejected = true
        }
        assertTrue(rejected)
    }

    @Test
    fun fallbackSquelchIncludesPositivePreEventWindow() {
        val decision = SquelchPolicy.decide(emptyList(), 1_000_000L)

        assertEquals(55_000L, decision.squelchMs)
        assertEquals(59_000L, decision.nextRecognitionDelayMs)
    }

    @Test
    fun tenRecentMissesUseRepeatedMissSquelch() {
        val nowMs = 1_000_000L
        val history = (0 until 10).map { index ->
            RecognitionHistoryEntry(
                elapsedRealtimeMs = nowMs - index * 10_000L,
                matched = false,
                shard = null,
                numericId = null,
                offsetsMs = emptyList(),
                durationMs = null,
            )
        }

        val decision = SquelchPolicy.decide(history, nowMs)

        assertEquals(90_000L, decision.squelchMs)
        assertEquals(94_000L, decision.nextRecognitionDelayMs)
    }

    @Test
    fun nearTrackEndUsesBoundedRemainingTime() {
        val nowMs = 1_000_000L
        val history = listOf(
            RecognitionHistoryEntry(
                elapsedRealtimeMs = nowMs,
                matched = true,
                shard = "USfixture",
                numericId = 1,
                offsetsMs = listOf(40_000L),
                durationMs = 60_000L,
            ),
        )

        val decision = SquelchPolicy.decide(history, nowMs)

        assertEquals(20_000L, decision.squelchMs)
        assertEquals(24_000L, decision.nextRecognitionDelayMs)
    }

    @Test
    fun onDemandSessionSuppressesOrganicRecognition() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        RecognitionSession.beginOnDemand().use {
            assertEquals(
                "an on-demand session is active",
                RecognitionGates.captureBlockReason(context),
            )
        }
    }

    @Test
    fun missingPhonePermissionSuppressesRecognition() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.adoptShellPermissionIdentity("android.permission.QUERY_USERS")
        try {
            val context = object : ContextWrapper(instrumentation.targetContext) {
                override fun checkSelfPermission(permission: String): Int =
                    if (permission == Manifest.permission.READ_PHONE_STATE) {
                        PackageManager.PERMISSION_DENIED
                    } else {
                        super.checkSelfPermission(permission)
                    }
            }
            assertEquals(
                "phone state is unavailable",
                RecognitionGates.captureBlockReason(context),
            )
        } finally {
            instrumentation.uiAutomation.dropShellPermissionIdentity()
        }
    }

    @Test
    fun unblockedDeviceStateAllowsCapture() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.adoptShellPermissionIdentity("android.permission.QUERY_USERS")
        try {
            assertEquals(
                null,
                RecognitionGates.captureBlockReason(instrumentation.targetContext),
            )
        } finally {
            instrumentation.uiAutomation.dropShellPermissionIdentity()
        }
    }

    @Test
    fun playbackClassifierRequiresExactMediaMusicStartedTuple() {
        assertTrue(
            RecognitionGates.isMusicPlayback(
                "com.example.player",
                AudioAttributes.USAGE_MEDIA,
                AudioAttributes.CONTENT_TYPE_MUSIC,
                AudioPlaybackConfiguration.PLAYER_STATE_STARTED,
            ),
        )
        assertFalse(
            RecognitionGates.isMusicPlayback(
                "com.example.player",
                AudioAttributes.USAGE_GAME,
                AudioAttributes.CONTENT_TYPE_MUSIC,
                AudioPlaybackConfiguration.PLAYER_STATE_STARTED,
            ),
        )
        assertFalse(
            RecognitionGates.isMusicPlayback(
                "com.example.player",
                AudioAttributes.USAGE_MEDIA,
                AudioAttributes.CONTENT_TYPE_SPEECH,
                AudioPlaybackConfiguration.PLAYER_STATE_STARTED,
            ),
        )
        assertFalse(
            RecognitionGates.isMusicPlayback(
                "com.example.player",
                AudioAttributes.USAGE_MEDIA,
                AudioAttributes.CONTENT_TYPE_MUSIC,
                AudioPlaybackConfiguration.PLAYER_STATE_PAUSED,
            ),
        )
    }

    @Test
    fun playbackIgnorelistUsesRecoveredPackagePrefixes() {
        for (packageName in listOf(
            "ak.alizandro.smartaudiobookplayer",
            "ak.alizandro.smartaudiobookplayer.pro",
            "com.breel.wallpapers.variant",
            "com.google.android.googlequicksearchbox",
        )) {
            assertFalse(
                RecognitionGates.isMusicPlayback(
                    packageName,
                    AudioAttributes.USAGE_MEDIA,
                    AudioAttributes.CONTENT_TYPE_MUSIC,
                    AudioPlaybackConfiguration.PLAYER_STATE_STARTED,
                ),
            )
        }
    }

    @Test
    fun privateStorageIsOwnerOnly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = PrivateStorage.deviceProtectedContext(context).filesDir

        assertEquals(448, Os.stat(directory.path).st_mode and 511)
    }

    @Test
    fun onlyEnabledForegroundSystemUserCanArmSoundTrigger() {
        assertTrue(SoundTriggerController.canArm(true, true, true))
        assertFalse(SoundTriggerController.canArm(true, false, true))
        assertFalse(SoundTriggerController.canArm(false, true, true))
        assertFalse(SoundTriggerController.canArm(true, true, false))
    }

    @Test
    fun rearmJobUsesTheSquelchDeadline() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val job = SoundTriggerRearmJob.jobInfo(context, 47_246L)

        assertEquals(47_246L, job.minLatencyMillis)
        assertEquals(47_246L, job.maxExecutionDelayMillis)
        assertEquals(ShardDownloadJobService::class.java.name, job.service.className)
        try {
            assertTrue(SoundTriggerRearmJob.schedule(context, 47_246L))
            val scheduler = checkNotNull(context.getSystemService(JobScheduler::class.java))
            assertEquals(job.service, scheduler.getPendingJob(SoundTriggerRearmJob.JOB_ID)?.service)
        } finally {
            SoundTriggerRearmJob.cancel(context)
        }
    }

    @Test
    fun incompleteCatalogRequestsImmediateBootSync() {
        assertTrue(
            BootReceiver.shouldRequestImmediateSync(Intent.ACTION_LOCKED_BOOT_COMPLETED, false),
        )
        assertTrue(BootReceiver.shouldRequestImmediateSync(Intent.ACTION_BOOT_COMPLETED, false))
        assertFalse(
            BootReceiver.shouldRequestImmediateSync(Intent.ACTION_LOCKED_BOOT_COMPLETED, true),
        )
        assertFalse(BootReceiver.shouldRequestImmediateSync(Intent.ACTION_USER_FOREGROUND, false))
    }

    @Test
    fun recognitionHistoryPersistsFieldsAndSupportsDeletion() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val databaseName = "recognition-history-test-${System.nanoTime()}.db"
        val protectedContext = context.createDeviceProtectedStorageContext()
        protectedContext.deleteDatabase(databaseName)
        try {
            val nowMs = System.currentTimeMillis()
            RecognitionHistoryStore(context, databaseName).use { store ->
                store.record(match("older", 1), nowMs - 1_000L)
                store.record(match("expired", 2), nowMs - RecognitionHistoryStore.RETENTION_MS - 1L)
                val newestId = store.record(match("newest", 3), nowMs)
                val entries = store.newestFirst()

                assertEquals(listOf("newest", "older"), entries.map(StoredRecognition::track))
                assertEquals("artist-newest", entries[0].artist)
                assertEquals("track-newest", entries[0].trackId)
                assertEquals("USnewest", entries[0].shard)
                assertEquals(3, entries[0].numericId)
                assertEquals(3.25, entries[0].offsetSeconds, 0.0)
                assertEquals(3.5, entries[0].score, 0.0)
                assertEquals(nowMs, entries[0].recognizedAtMs)
                assertTrue(store.delete(newestId))
                assertEquals(listOf("older"), store.newestFirst().map(StoredRecognition::track))
                assertEquals(1, store.clear())
                assertTrue(store.newestFirst().isEmpty())
                repeat(RecognitionHistoryStore.MAX_ENTRIES + 1) { index ->
                    store.record(match("bounded-$index", index), nowMs + index)
                }
                val bounded = store.newestFirst()
                assertEquals(RecognitionHistoryStore.MAX_ENTRIES, bounded.size)
                assertEquals("bounded-1000", bounded.first().track)
                assertEquals("bounded-1", bounded.last().track)
                assertEquals(RecognitionHistoryStore.MAX_ENTRIES, store.clear())
            }
            RecognitionHistoryStore(context, databaseName).use { store ->
                assertTrue(store.newestFirst().isEmpty())
            }
        } finally {
            protectedContext.deleteDatabase(databaseName)
        }
    }

    @Test
    fun directBootHistorySeedIsControlledByInstrumentationArgument() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val store = RecognitionHistoryStore(instrumentation.targetContext)
        store.use {
            it.newestFirst()
                .filter { entry -> entry.trackId == DIRECT_BOOT_TRACK_ID }
                .forEach { entry -> it.delete(entry.id) }
            if (InstrumentationRegistry.getArguments().getString("seed_history") == "true") {
                it.record(match("direct-boot-proof", 404))
                assertEquals(
                    1,
                    it.newestFirst().count { entry -> entry.trackId == DIRECT_BOOT_TRACK_ID },
                )
            }
        }
    }

    private fun match(name: String, numericId: Int) = CatalogMatch(
        title = name,
        artist = "artist-$name",
        trackId = if (name == "direct-boot-proof") DIRECT_BOOT_TRACK_ID else "track-$name",
        numericId = numericId,
        shard = "US$name",
        score = numericId + 0.5,
        offsetSeconds = numericId + 0.25,
        durationSeconds = null,
    )

    private companion object {
        const val DIRECT_BOOT_TRACK_ID = "stage4-direct-boot-proof"
    }
}
