#!/usr/bin/env python3

import argparse
import hashlib
import struct
import zipfile
from pathlib import Path


APK_SHA256 = "9aee83de6061dbfb853c0a8f3120006db37e9a12aecb718a1b0faca8999127ee"
LIBRARY_SHA256 = "fed8dc4ee6eed9741b958ca97b03fa450ca99712bb201ce1d28282a280e36342"
CONFIG_SHA256 = "b55bde286a788ac53b29f9a89e07a74171eb2e83fb47e51a9d5c8c6d305434d7"
HUFFMAN_SHA256 = "ede41598fe067533a87afff31a4d062ef856905efaa51a89f7ae173819d0f53d"
WEIGHTS_SHA256 = "4bd2654a980fcdfc6aa4e8ebf3fd0d4c0e21fb36bb7ebf227e1551fd359f2c0c"

MUSIC_RTTI = b"N10audio_ears21RecognizerMatchResultE"
EMBEDDER_RTTI = b"N10audio_ears24SoundSearchFingerprinterE"
MUSIC_NETWORK_SIZE = 0x4B0D4
EMBEDDER_NETWORK_SIZE = 0x363014
OUTPUT_SIZE = 0x3CF7FC
MUSIC_DESTINATION = 0x20C40
EMBEDDER_DESTINATION = 0x6C7E8
HUFFMAN_DESTINATION = 0x1F198
HUFFMAN_SIZE = 0xC1A
HUFFMAN_SENTINEL = bytes.fromhex(
    "0000000801000020030000281d0000280d000028150000280500002819000028"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_hash(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} SHA-256 is {actual}, expected {expected}")


def unique_offset(data: bytes, needle: bytes, name: str) -> int:
    offset = data.find(needle)
    if offset < 0:
        raise ValueError(f"{name} anchor is absent")
    if data.find(needle, offset + 1) >= 0:
        raise ValueError(f"{name} anchor is not unique")
    return offset


def archive_member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"found {len(names)} archive members ending in {suffix}")
    return archive.read(names[0])


def config_member(archive: zipfile.ZipFile) -> bytes:
    matches = []
    for info in archive.infolist():
        if info.file_size != 500_745:
            continue
        data = archive.read(info)
        if sha256(data) == CONFIG_SHA256:
            matches.append(data)
    if len(matches) != 1:
        raise ValueError(f"found {len(matches)} matching config resources")
    return matches[0]


def carve_weights(library: bytes) -> bytes:
    music_end = unique_offset(library, MUSIC_RTTI, "music network")
    embedder_end = unique_offset(library, EMBEDDER_RTTI, "embedder network")
    music_start = music_end - MUSIC_NETWORK_SIZE
    embedder_start = embedder_end - EMBEDDER_NETWORK_SIZE
    if music_start < 0 or embedder_start < 0:
        raise ValueError("an RTTI anchor appears before its network")
    if struct.unpack_from("<4I", library, music_start) != (3, 3, 1, 8):
        raise ValueError("music network begins with an unexpected shape")
    if struct.unpack_from("<8I", library, embedder_start) != (1, 42, 512, 1, 1, 1, 16, 2):
        raise ValueError("embedder network begins with an unexpected shape")

    huffman_start = unique_offset(library, HUFFMAN_SENTINEL, "Huffman table")
    huffman = library[huffman_start : huffman_start + HUFFMAN_SIZE]
    if len(huffman) != HUFFMAN_SIZE:
        raise ValueError("Huffman table is truncated")
    require_hash("Huffman table", sha256(huffman), HUFFMAN_SHA256)

    output = bytearray(OUTPUT_SIZE)
    output[HUFFMAN_DESTINATION : HUFFMAN_DESTINATION + HUFFMAN_SIZE] = huffman
    output[MUSIC_DESTINATION : MUSIC_DESTINATION + MUSIC_NETWORK_SIZE] = library[
        music_start:music_end
    ]
    output[EMBEDDER_DESTINATION : EMBEDDER_DESTINATION + EMBEDDER_NETWORK_SIZE] = library[
        embedder_start:embedder_end
    ]
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("apk", type=Path)
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args()

    require_hash("ASI APK", file_sha256(arguments.apk), APK_SHA256)
    with zipfile.ZipFile(arguments.apk) as archive:
        library = archive_member(archive, "/arm64-v8a/libsense_nnfp_v3.so")
        config = config_member(archive)
    require_hash("NNFP library", sha256(library), LIBRARY_SHA256)

    assets = arguments.repository / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    weights_path = assets / "nnfp_v3.weights"
    config_path = assets / "v3_config_tah.pb"
    weights = carve_weights(library)
    require_hash("carved weights", sha256(weights), WEIGHTS_SHA256)
    weights_path.write_bytes(weights)
    config_path.write_bytes(config)
    print(f"wrote {weights_path} sha256 {file_sha256(weights_path)}")
    print(f"wrote {config_path} sha256 {file_sha256(config_path)}")


if __name__ == "__main__":
    main()
