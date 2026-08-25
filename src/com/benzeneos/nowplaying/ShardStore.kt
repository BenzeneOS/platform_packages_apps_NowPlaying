package com.benzeneos.nowplaying

import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import org.json.JSONArray
import org.json.JSONObject

data class ActivePack(val name: String, val size: Long)

data class ActiveCatalog(
    val build: String,
    val country: String,
    val packs: List<ActivePack>,
    val complete: Boolean,
)

data class CatalogSyncResult(
    val build: String,
    val country: String,
    val shardCount: Int,
    val catalogBytes: Long,
    val downloadedBytes: Long,
    val elapsedMs: Long,
)

class ShardStore(context: Context) {
    private val protectedContext = PrivateStorage.deviceProtectedContext(context)
    private val root = PrivateStorage.ensureDirectory(File(protectedContext.filesDir, "catalog"))
    private val objects = PrivateStorage.ensureDirectory(File(root, "objects"))
    private val activeFile = File(root, "active.json")

    fun readActive(): ActiveCatalog? = try {
        if (!activeFile.isFile || activeFile.length() !in 1..MAX_ACTIVE_BYTES) {
            null
        } else {
            parseActive(activeFile.readText(Charsets.UTF_8)).takeIf(::allFilesMatch)
        }
    } catch (error: Throwable) {
        Log.e(TAG, "active catalog metadata was rejected", error)
        null
    }

    fun activeFiles(): List<File> = readActive()?.packs?.map { pack ->
        File(objects, pack.name)
    } ?: emptyList()

    fun isComplete(): Boolean = readActive()?.complete == true

    fun sync(
        source: CatalogSource,
        country: String,
        storageCapBytes: Long?,
        onProgress: (ActiveCatalog, Long) -> Unit,
    ): CatalogSyncResult = synchronized(STORE_LOCK) {
        syncLocked(source, country, storageCapBytes, onProgress)
    }

    fun deleteDownloaded(): Pair<Int, Long> = synchronized(STORE_LOCK) {
        NativeRecognizer.dropDownloadedShards()
        val files = objects.listFiles()?.filter(File::isFile) ?: emptyList()
        val bytes = files.sumOf(File::length)
        files.forEach { file -> Files.deleteIfExists(file.toPath()) }
        Files.deleteIfExists(activeFile.toPath())
        Files.deleteIfExists(File(root, "active.json.partial").toPath())
        files.size to bytes
    }

    private fun syncLocked(
        source: CatalogSource,
        country: String,
        storageCapBytes: Long?,
        onProgress: (ActiveCatalog, Long) -> Unit,
    ): CatalogSyncResult {
        require(COUNTRY.matches(country)) { "Device country code is invalid" }
        val started = SystemClock.elapsedRealtime()
        objects.listFiles()
            ?.filter { file -> file.name.endsWith(".partial") }
            ?.forEach { file -> Files.deleteIfExists(file.toPath()) }
        NativeRecognizer.initialize(protectedContext)

        val build = source.latestBuild()
        val desired = source.manifest(build).select(country, storageCapBytes)
        val desiredActive = desired.map { pack -> ActivePack(pack.name, pack.size) }
        var active = readActive()
        var downloadedBytes = 0L

        if (active?.build == build && active.country == country && active.packs == desiredActive) {
            if (!active.complete) {
                active = active.copy(complete = true)
                activate(active)
            }
            prune(desiredActive)
            return result(active, downloadedBytes, started)
        }

        val activePrefix = active?.build == build &&
            active.country == country &&
            desiredActive.take(active.packs.size) == active.packs
        val incremental = active == null || activePrefix
        if (incremental) {
            val completed = active?.packs?.toMutableList() ?: ArrayList()
            for (index in completed.size until desired.size) {
                val pack = desired[index]
                downloadedBytes = Math.addExact(
                    downloadedBytes,
                    ensureObject(source, build, pack, active),
                )
                completed += ActivePack(pack.name, pack.size)
                active = ActiveCatalog(
                    build,
                    country,
                    completed.toList(),
                    complete = completed.size == desired.size,
                )
                activate(active)
                onProgress(active, downloadedBytes)
            }
            if (completed.size > desired.size) {
                active = ActiveCatalog(build, country, desiredActive, complete = true)
                activate(active)
            }
            if (active == null) {
                active = ActiveCatalog(build, country, emptyList(), complete = true)
                activate(active)
            }
        } else {
            for (pack in desired) {
                downloadedBytes = Math.addExact(
                    downloadedBytes,
                    ensureObject(source, build, pack, active),
                )
                onProgress(
                    ActiveCatalog(build, country, desiredActive, complete = false),
                    downloadedBytes,
                )
            }
            active = ActiveCatalog(build, country, desiredActive, complete = true)
            activate(active)
        }
        prune(active.packs)
        return result(active, downloadedBytes, started)
    }

    private fun ensureObject(
        source: CatalogSource,
        build: String,
        pack: CatalogPack,
        active: ActiveCatalog?,
    ): Long {
        val destination = File(objects, pack.name)
        val alreadyActive = active?.packs?.contains(ActivePack(pack.name, pack.size)) == true
        if (destination.isFile && destination.length() == pack.size) {
            if (!alreadyActive) {
                try {
                    NativeRecognizer.validateShard(destination)
                } catch (error: Throwable) {
                    Files.deleteIfExists(destination.toPath())
                    Log.e(TAG, "cached catalog shard ${pack.name} was rejected", error)
                }
            }
            if (destination.isFile) {
                return 0L
            }
        } else {
            Files.deleteIfExists(destination.toPath())
        }

        source.download(build, pack, destination)
        try {
            NativeRecognizer.validateShard(destination)
        } catch (error: Throwable) {
            Files.deleteIfExists(destination.toPath())
            throw IOException("Downloaded catalog shard ${pack.name} was rejected", error)
        }
        return pack.size
    }

    private fun activate(active: ActiveCatalog) {
        val files = active.packs.map { pack -> File(objects, pack.name) }
        check(files.zip(active.packs).all { (file, pack) ->
            file.isFile && file.length() == pack.size
        }) { "Active catalog contains a missing or truncated shard" }
        NativeRecognizer.replaceShards(files)
        writeActive(active)
        Log.i(TAG, "catalog activation build ${active.build} shards ${active.packs.size}")
    }

    private fun writeActive(active: ActiveCatalog) {
        val temporary = File(root, "active.json.partial")
        val packs = JSONArray()
        active.packs.forEach { pack ->
            packs.put(JSONObject().put("name", pack.name).put("size", pack.size))
        }
        val encoded = JSONObject()
            .put("build", active.build)
            .put("country", active.country)
            .put("packs", packs)
            .put("complete", active.complete)
            .toString()
        try {
            FileOutputStream(temporary).use { output ->
                output.write(encoded.toByteArray(Charsets.UTF_8))
                output.fd.sync()
            }
            Files.move(
                temporary.toPath(),
                activeFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } finally {
            Files.deleteIfExists(temporary.toPath())
        }
    }

    private fun parseActive(encoded: String): ActiveCatalog {
        val root = JSONObject(encoded)
        val build = root.getString("build")
        val country = root.getString("country")
        require(BUILD.matches(build)) { "Active catalog build name is invalid" }
        require(COUNTRY.matches(country)) { "Active catalog country is invalid" }
        val values = root.getJSONArray("packs")
        require(values.length() <= MAX_ACTIVE_PACKS) { "Active catalog has too many shards" }
        val names = HashSet<String>()
        val packs = ArrayList<ActivePack>()
        for (index in 0 until values.length()) {
            val value = values.getJSONObject(index)
            val name = value.getString("name")
            val size = value.getLong("size")
            require(SHARD.matches(name) && names.add(name)) { "Active shard name is invalid" }
            require(size in 1..MAX_PACK_BYTES) { "Active shard size is invalid" }
            packs += ActivePack(name, size)
        }
        return ActiveCatalog(build, country, packs, root.optBoolean("complete", false))
    }

    private fun allFilesMatch(active: ActiveCatalog): Boolean = active.packs.all { pack ->
        val file = File(objects, pack.name)
        file.isFile && file.length() == pack.size
    }

    private fun prune(keep: List<ActivePack>) {
        val names = keep.mapTo(HashSet(), ActivePack::name)
        objects.listFiles()?.forEach { file ->
            if (file.isFile && file.name !in names) {
                Files.deleteIfExists(file.toPath())
            }
        }
    }

    private fun result(
        active: ActiveCatalog,
        downloadedBytes: Long,
        started: Long,
    ) = CatalogSyncResult(
        build = active.build,
        country = active.country,
        shardCount = active.packs.size,
        catalogBytes = active.packs.sumOf(ActivePack::size),
        downloadedBytes = downloadedBytes,
        elapsedMs = SystemClock.elapsedRealtime() - started,
    )

    private companion object {
        val STORE_LOCK = Any()
        const val TAG = "NowPlayingCatalog"
        const val MAX_ACTIVE_BYTES = 1_048_576L
        const val MAX_ACTIVE_PACKS = 10_000
        const val MAX_PACK_BYTES = 134_217_728L
        val BUILD = Regex("[0-9]{8}-[0-9]{6}")
        val COUNTRY = Regex("[a-z]{2}")
        val SHARD = Regex("[A-Z]{2}[0-9a-f]{32}")
    }
}
