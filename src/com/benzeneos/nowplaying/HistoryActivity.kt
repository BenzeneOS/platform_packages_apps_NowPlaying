package com.benzeneos.nowplaying

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Typeface
import android.os.Bundle
import android.text.format.DateFormat
import android.text.format.DateUtils
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.Date
import java.util.concurrent.Executors

class HistoryActivity : Activity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var clearButton: Button
    private lateinit var entriesView: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.history_activity_label)

        clearButton = Button(this).apply {
            text = getString(R.string.action_clear_all)
            isEnabled = false
            setOnClickListener { confirmClear() }
        }
        entriesView = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val header = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            orientation = LinearLayout.HORIZONTAL
            addView(
                TextView(this@HistoryActivity).apply {
                    text = getString(R.string.history_heading)
                    textSize = 22f
                    setTypeface(typeface, Typeface.BOLD)
                },
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(clearButton)
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(12), dp(20), dp(24))
            addView(header, matchWidth())
            addView(entriesView, matchWidth(topMargin = dp(8)))
        }
        setContentView(ScrollView(this).apply { addView(content) })
    }

    override fun onResume() {
        super.onResume()
        loadEntries()
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }

    private fun loadEntries() {
        executor.execute {
            val entries = RecognitionHistoryStore(this).use { store -> store.newestFirst() }
            runOnUiThread {
                if (!isDestroyed) {
                    render(entries)
                }
            }
        }
    }

    private fun render(entries: List<StoredRecognition>) {
        clearButton.isEnabled = entries.isNotEmpty()
        entriesView.removeAllViews()
        if (entries.isEmpty()) {
            entriesView.addView(
                TextView(this).apply {
                    text = getString(R.string.history_empty)
                    textSize = 16f
                    setPadding(0, dp(24), 0, 0)
                },
            )
            return
        }
        entries.forEachIndexed { index, entry ->
            if (index != 0) {
                entriesView.addView(View(this).apply { setBackgroundColor(0x33777777) }, separator())
            }
            entriesView.addView(historyRow(entry), matchWidth())
        }
    }

    private fun historyRow(entry: StoredRecognition): View {
        val details = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(
                TextView(this@HistoryActivity).apply {
                    text = entry.track
                    textSize = 18f
                    setTypeface(typeface, Typeface.BOLD)
                },
            )
            addView(TextView(this@HistoryActivity).apply {
                text = entry.artist
                textSize = 16f
            })
            addView(TextView(this@HistoryActivity).apply {
                text = formatTime(entry.recognizedAtMs)
                textSize = 14f
            })
        }
        return LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(12), 0, dp(12))
            addView(details, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(Button(this@HistoryActivity).apply {
                text = getString(R.string.action_delete)
                setOnClickListener { delete(entry.id) }
            })
        }
    }

    private fun formatTime(timestampMs: Long): String {
        val relative = DateUtils.getRelativeTimeSpanString(
            timestampMs,
            System.currentTimeMillis(),
            DateUtils.MINUTE_IN_MILLIS,
        )
        val date = Date(timestampMs)
        val absolute = DateFormat.getMediumDateFormat(this).format(date) +
            " " + DateFormat.getTimeFormat(this).format(date)
        return getString(R.string.history_timestamp, relative, absolute)
    }

    private fun delete(id: Long) {
        executor.execute {
            RecognitionHistoryStore(this).use { store -> store.delete(id) }
            val entries = RecognitionHistoryStore(this).use { store -> store.newestFirst() }
            runOnUiThread {
                if (!isDestroyed) {
                    render(entries)
                }
            }
        }
    }

    private fun confirmClear() {
        AlertDialog.Builder(this)
            .setTitle(R.string.clear_dialog_title)
            .setMessage(R.string.clear_dialog_message)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.clear_dialog_confirm) { _, _ -> clear() }
            .show()
    }

    private fun clear() {
        executor.execute {
            RecognitionHistoryStore(this).use { store -> store.clear() }
            runOnUiThread {
                if (!isDestroyed) {
                    render(emptyList())
                }
            }
        }
    }

    private fun matchWidth(topMargin: Int = 0) = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.WRAP_CONTENT,
    ).apply {
        this.topMargin = topMargin
    }

    private fun separator() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        dp(1),
    )

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
