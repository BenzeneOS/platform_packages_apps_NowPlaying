package com.benzeneos.nowplaying

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.Closeable

data class StoredRecognition(
    val id: Long,
    val track: String,
    val artist: String,
    val trackId: String,
    val shard: String,
    val numericId: Int,
    val offsetSeconds: Double,
    val score: Double,
    val recognizedAtMs: Long,
)

class RecognitionHistoryStore(
    context: Context,
    databaseName: String = DATABASE_NAME,
) : Closeable {
    private val helper = HistoryDatabase(
        context.createDeviceProtectedStorageContext(),
        databaseName,
    )

    fun record(
        match: CatalogMatch,
        recognizedAtMs: Long = System.currentTimeMillis(),
    ): Long {
        val database = helper.writableDatabase
        database.beginTransaction()
        return try {
            val id = database.insertOrThrow(
                TABLE,
                null,
                ContentValues().apply {
                    put(TRACK, match.title)
                    put(ARTIST, match.artist)
                    put(TRACK_ID, match.trackId)
                    put(SHARD, match.shard)
                    put(NUMERIC_ID, match.numericId)
                    put(OFFSET_SECONDS, match.offsetSeconds)
                    put(SCORE, match.score)
                    put(RECOGNIZED_AT_MS, recognizedAtMs)
                },
            )
            prune(database, recognizedAtMs)
            database.setTransactionSuccessful()
            id
        } finally {
            database.endTransaction()
        }
    }

    fun newestFirst(): List<StoredRecognition> {
        val entries = ArrayList<StoredRecognition>()
        helper.readableDatabase.query(
            TABLE,
            COLUMNS,
            null,
            null,
            null,
            null,
            "$RECOGNIZED_AT_MS DESC, $ID DESC",
            MAX_ENTRIES.toString(),
        ).use { cursor ->
            while (cursor.moveToNext()) {
                entries += StoredRecognition(
                    id = cursor.getLong(0),
                    track = cursor.getString(1),
                    artist = cursor.getString(2),
                    trackId = cursor.getString(3),
                    shard = cursor.getString(4),
                    numericId = cursor.getInt(5),
                    offsetSeconds = cursor.getDouble(6),
                    score = cursor.getDouble(7),
                    recognizedAtMs = cursor.getLong(8),
                )
            }
        }
        return entries
    }

    fun delete(id: Long): Boolean = helper.writableDatabase.delete(
        TABLE,
        "$ID = ?",
        arrayOf(id.toString()),
    ) == 1

    fun clear(): Int = helper.writableDatabase.delete(TABLE, null, null)

    override fun close() {
        helper.close()
    }

    private fun prune(database: SQLiteDatabase, nowMs: Long) {
        database.delete(
            TABLE,
            "$RECOGNIZED_AT_MS < ?",
            arrayOf((nowMs - RETENTION_MS).toString()),
        )
        database.execSQL(
            "DELETE FROM $TABLE WHERE $ID IN " +
                "(SELECT $ID FROM $TABLE ORDER BY $RECOGNIZED_AT_MS DESC, $ID DESC " +
                "LIMIT -1 OFFSET $MAX_ENTRIES)",
        )
    }

    private class HistoryDatabase(context: Context, name: String) :
        SQLiteOpenHelper(context, name, null, DATABASE_VERSION) {
        override fun onCreate(database: SQLiteDatabase) {
            database.execSQL(
                "CREATE TABLE $TABLE (" +
                    "$ID INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "$TRACK TEXT NOT NULL, " +
                    "$ARTIST TEXT NOT NULL, " +
                    "$TRACK_ID TEXT NOT NULL, " +
                    "$SHARD TEXT NOT NULL, " +
                    "$NUMERIC_ID INTEGER NOT NULL, " +
                    "$OFFSET_SECONDS REAL NOT NULL, " +
                    "$SCORE REAL NOT NULL, " +
                    "$RECOGNIZED_AT_MS INTEGER NOT NULL)",
            )
            database.execSQL(
                "CREATE INDEX recognition_history_time ON $TABLE " +
                    "($RECOGNIZED_AT_MS DESC, $ID DESC)",
            )
        }

        override fun onUpgrade(database: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
    }

    companion object {
        const val DATABASE_NAME = "recognition_history.db"
        const val MAX_ENTRIES = 1_000
        const val RETENTION_MS = 2_592_000_000L

        private const val DATABASE_VERSION = 1
        private const val TABLE = "recognition_history"
        private const val ID = "id"
        private const val TRACK = "track"
        private const val ARTIST = "artist"
        private const val TRACK_ID = "track_id"
        private const val SHARD = "shard"
        private const val NUMERIC_ID = "numeric_id"
        private const val OFFSET_SECONDS = "offset_seconds"
        private const val SCORE = "score"
        private const val RECOGNIZED_AT_MS = "recognized_at_ms"
        private val COLUMNS = arrayOf(
            ID,
            TRACK,
            ARTIST,
            TRACK_ID,
            SHARD,
            NUMERIC_ID,
            OFFSET_SECONDS,
            SCORE,
            RECOGNIZED_AT_MS,
        )
    }
}
