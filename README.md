# Now Playing

Ambient music recognition for BenzeneOS. The phone identifies music playing around it entirely
on device, with no network at recognition time and no Google code running anywhere in the
stack.

This is a clean-room implementation. Google's model weights and search parameters ship as
data. Every line of executing code is ours.

## How it works

The application processor never records continuously. That is what makes the feature
affordable rather than a battery problem.

```
vendor DSP     always-on 23 KB music detector
    |          one-shot SoundTrigger event plus a capture session
    v
AudioCapture   four to eight seconds of already-buffered PCM16
    |
    v
frontend       resample 22,050 to 11,025 Hz, periodic Hamming window,
    |          1,024-point FFT at hop 512, magnitude, DC dropped,
    |          512 bins across 42 frames
    v
embedder       35 tensors, 887,712 float32 parameters
    |          96-dim L2-normalized embedding, one per second of audio
    v
search         IVFADC, nearest 20 of 1,000 partitions per shard,
    |          12-byte product codes over 12 subquantizers
    v
scorer         softplus aggregate over per-query weights,
    |          accepted when the score is strictly greater than zero
    v
history        private track, artist, match details, and recognition time
```

A music gate runs ahead of the catalog search, so speech and ambient noise are rejected before
anything touches the index.

## Status

The launcher opens device-local recognition history with individual deletion and clear-all.
History remains available before first unlock and keeps at most 30 days or 1,000 entries.
Regional catalog shards download on unmetered networks into device-protected private storage.
The lockscreen surface and Settings integration remain future work.

## Parity

Every stage is verified against Google's own implementation rather than against a
specification. The spectrogram matches bit for bit across all 21,504 model inputs. The
embedder reproduces Google's output on a fixed input. The scorer reproduces Google's
aggregate to within `2e-6` when fed the same embeddings. End to end, on identical PCM, this
implementation accepts the same track at the same offset that Google accepts.

Recognition correctness is measured as agreement with Google's recognizer, never against a
filename. A downloaded track labelled with a title is not evidence of what the audio contains.

## Assets

Five data files, no executable bytes.

| File | Source |
| --- | --- |
| `assets/nnfp_v3.weights` | Carved from ASI, both networks at their native offsets |
| `assets/v3_config_tah.pb` | ASI resource, partition centroids and the PQ codebook |
| `matcher_tah.leveldb` | Vendor blob, the core index |
| `music_detector.sound_model` | Vendor blob, the DSP wake model |
| `music_detector.descriptor` | Vendor blob, the DSP classifier field schema |

The last three arrive through the ordinary vendor pipeline. Country shards are published by
Google in a public bucket and are fetched at runtime, so no catalog data is stored here.

## Rebuilding the checked-in assets

The two checked-in files were recovered from the Google ASI build below.

`DevicePersonalizationPrebuiltPixel2024-bfinal_aiai_20250217.00_RC08`

The source APK SHA-256 is

`9aee83de6061dbfb853c0a8f3120006db37e9a12aecb718a1b0faca8999127ee`

The carved weights SHA-256 is

`4bd2654a980fcdfc6aa4e8ebf3fd0d4c0e21fb36bb7ebf227e1551fd359f2c0c`

The config SHA-256 is

`b55bde286a788ac53b29f9a89e07a74171eb2e83fb47e51a9d5c8c6d305434d7`

Run the carve tool with the APK and this repository as its inputs.

```sh
python3 tools/carve_assets.py /path/to/DevicePersonalizationPrebuilt.apk .
```

The tool verifies the source build, finds both network boundaries from their RTTI names,
validates the Huffman table, and writes the same offset-preserving data image consumed by the
Rust core. It anchors on the RTTI symbols rather than on fixed offsets, so a different ASI
build fails loudly instead of carving the wrong bytes.

## Layout

```
rust/core/    recognizer, no dependencies outside std
rust/jni/     JNI shim
src/          Kotlin SoundTrigger wiring and capture
assets/       carved model data
tools/        carve tool
```
