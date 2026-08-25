package com.benzeneos.nowplaying

import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InterruptedIOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Locale
import org.json.JSONObject

data class CatalogPack(
    val name: String,
    val size: Long,
    val compressedSize: Long,
    val countryCodes: Set<String>,
    val downloadUrls: List<String>,
)

data class CatalogManifest(
    val build: String,
    val packs: List<CatalogPack>,
) {
    fun select(country: String, storageCapBytes: Long?): List<CatalogPack> {
        val selected = ArrayList<CatalogPack>()
        var total = 0L
        for (pack in packs) {
            if (country !in pack.countryCodes) {
                continue
            }
            val next = Math.addExact(total, pack.size)
            if (storageCapBytes != null && next > storageCapBytes) {
                break
            }
            selected += pack
            total = next
        }
        return selected
    }
}

object CatalogManifestParser {
    private val buildName = Regex("[0-9]{8}-[0-9]{6}")
    private val shardName = Regex("[A-Z]{2}[0-9a-f]{32}")
    private const val MAX_PACKS = 10_000
    private const val MAX_PACK_BYTES = 134_217_728L
    private const val MAX_DOWNLOAD_URLS = 8

    fun parse(build: String, encoded: String): CatalogManifest {
        require(buildName.matches(build)) { "Catalog build name is invalid" }
        val values = JSONObject(encoded).getJSONArray("packs")
        require(values.length() <= MAX_PACKS) { "Catalog manifest has too many packs" }
        val names = HashSet<String>()
        val packs = ArrayList<CatalogPack>()
        for (index in 0 until values.length()) {
            val value = values.getJSONObject(index)
            if (value.getString("extra_type") != "shard") {
                continue
            }
            val name = value.getString("name")
            require(shardName.matches(name)) { "Catalog shard name is invalid" }
            require(names.add(name)) { "Catalog manifest repeats a shard name" }
            val size = value.getLong("size")
            val compressedSize = value.getLong("compressed_size")
            require(size in 1..MAX_PACK_BYTES) { "Catalog shard size is invalid" }
            require(compressedSize in 1..MAX_PACK_BYTES) {
                "Catalog compressed shard size is invalid"
            }
            val urls = value.getJSONArray("download_urls")
            require(urls.length() in 1..MAX_DOWNLOAD_URLS) {
                "Catalog shard has an invalid download URL count"
            }
            val downloadUrls = ArrayList<String>(urls.length())
            for (urlIndex in 0 until urls.length()) {
                val url = urls.getString(urlIndex)
                require(URL(url).protocol == "https") { "Catalog shard URL must use HTTPS" }
                downloadUrls += url
            }
            val countryCodes = value.getString("extra_country_codes")
                .split(',')
                .map(String::trim)
                .filter(String::isNotEmpty)
                .map { code -> code.lowercase(Locale.ROOT) }
                .toSet()
            require(countryCodes.isNotEmpty()) { "Catalog shard has no country code" }
            require(countryCodes.all { code -> code.matches(Regex("[a-z]{2}")) }) {
                "Catalog shard has an invalid country code"
            }
            packs += CatalogPack(name, size, compressedSize, countryCodes, downloadUrls)
        }
        return CatalogManifest(build, packs)
    }
}

class CatalogSource(private val baseUrl: String) {
    private val normalizedBase = CatalogSettings.normalizeBaseUrl(baseUrl)

    fun latestBuild(): String {
        var token: String? = null
        var latest: String? = null
        repeat(MAX_LISTING_PAGES) {
            val builder = Uri.parse("$normalizedBase/storage/v1/b/music-iq-db/o")
                .buildUpon()
                .appendQueryParameter("prefix", "updatable_db_v3/")
                .appendQueryParameter("delimiter", "/")
                .appendQueryParameter("maxResults", "1000")
            if (token != null) {
                builder.appendQueryParameter("pageToken", token)
            }
            val root = JSONObject(getText(builder.build().toString(), MAX_LISTING_BYTES))
            val prefixes = root.optJSONArray("prefixes")
            if (prefixes != null) {
                for (index in 0 until prefixes.length()) {
                    val prefix = prefixes.getString(index)
                    val build = prefix.removePrefix("updatable_db_v3/").removeSuffix("/")
                    if (BUILD_NAME.matches(build) && (latest == null || build > latest)) {
                        latest = build
                    }
                }
            }
            token = root.optString("nextPageToken").ifEmpty { null }
            if (token == null) {
                return latest ?: throw IOException("Catalog listing contains no builds")
            }
        }
        throw IOException("Catalog listing exceeds the page limit")
    }

    fun manifest(build: String): CatalogManifest {
        require(BUILD_NAME.matches(build)) { "Catalog build name is invalid" }
        val url = "$normalizedBase/music-iq-db/updatable_db_v3/$build/manifest.json"
        return CatalogManifestParser.parse(build, getText(url, MAX_MANIFEST_BYTES))
    }

    fun download(build: String, pack: CatalogPack, destination: File) {
        require(BUILD_NAME.matches(build)) { "Catalog build name is invalid" }
        val url = if (normalizedBase == CatalogSettings.DEFAULT_BASE_URL) {
            pack.downloadUrls.first()
        } else {
            "$normalizedBase/music-iq-db/updatable_db_v3/$build/${pack.name}"
        }
        val temporary = File(destination.parentFile, "${pack.name}.partial")
        Files.deleteIfExists(temporary.toPath())
        try {
            withConnection(url) { connection ->
                val declared = connection.contentLengthLong
                if (declared >= 0 && declared != pack.size) {
                    throw IOException("Catalog shard Content-Length does not match its manifest")
                }
                FileOutputStream(temporary).use { output ->
                    connection.inputStream.use { input ->
                        val buffer = ByteArray(64 * 1024)
                        var written = 0L
                        while (true) {
                            if (Thread.currentThread().isInterrupted) {
                                throw InterruptedIOException("Catalog download was interrupted")
                            }
                            val count = input.read(buffer)
                            if (count < 0) {
                                break
                            }
                            written = Math.addExact(written, count.toLong())
                            if (written > pack.size) {
                                throw IOException("Catalog shard is larger than its manifest size")
                            }
                            output.write(buffer, 0, count)
                        }
                        if (written != pack.size) {
                            throw IOException("Catalog shard is truncated")
                        }
                    }
                    output.fd.sync()
                }
            }
            Files.move(
                temporary.toPath(),
                destination.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } finally {
            Files.deleteIfExists(temporary.toPath())
        }
    }

    private fun getText(url: String, limit: Int): String {
        val bytes = withConnection(url) { connection ->
            val declared = connection.contentLengthLong
            if (declared > limit) {
                throw IOException("Catalog response exceeds its size limit")
            }
            val output = ByteArrayOutputStream(declared.coerceAtLeast(0).coerceAtMost(limit.toLong()).toInt())
            connection.inputStream.use { input ->
                val buffer = ByteArray(16 * 1024)
                var total = 0
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) {
                        break
                    }
                    total = Math.addExact(total, count)
                    if (total > limit) {
                        throw IOException("Catalog response exceeds its size limit")
                    }
                    output.write(buffer, 0, count)
                }
            }
            output.toByteArray()
        }
        return Charsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    }

    private fun <T> withConnection(url: String, operation: (HttpURLConnection) -> T): T {
        var current = URL(url)
        repeat(MAX_REDIRECTS + 1) { redirect ->
            require(current.protocol == "https") { "Catalog URL must use HTTPS" }
            val connection = current.openConnection() as HttpURLConnection
            connection.connectTimeout = CONNECT_TIMEOUT_MS
            connection.readTimeout = READ_TIMEOUT_MS
            connection.instanceFollowRedirects = false
            connection.setRequestProperty("Accept-Encoding", "identity")
            connection.setRequestProperty("User-Agent", "NowPlayingCatalog/1")
            val response = connection.responseCode
            if (response in 300..399) {
                val location = connection.getHeaderField("Location")
                    ?: throw IOException("Catalog redirect has no location")
                connection.disconnect()
                if (redirect == MAX_REDIRECTS) {
                    throw IOException("Catalog request has too many redirects")
                }
                current = URL(current, location)
            } else {
                if (response != HttpURLConnection.HTTP_OK) {
                    connection.disconnect()
                    throw IOException("Catalog request returned HTTP $response")
                }
                try {
                    return operation(connection)
                } finally {
                    connection.disconnect()
                }
            }
        }
        throw IOException("Catalog request has too many redirects")
    }

    private companion object {
        val BUILD_NAME = Regex("[0-9]{8}-[0-9]{6}")
        const val CONNECT_TIMEOUT_MS = 20_000
        const val READ_TIMEOUT_MS = 30_000
        const val MAX_REDIRECTS = 5
        const val MAX_LISTING_PAGES = 32
        const val MAX_LISTING_BYTES = 2 * 1024 * 1024
        const val MAX_MANIFEST_BYTES = 4 * 1024 * 1024
    }
}
