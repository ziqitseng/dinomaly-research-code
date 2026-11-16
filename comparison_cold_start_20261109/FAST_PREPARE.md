# Faster preparation and restart

Preparation now defaults to four concurrent CPU/IO worker threads. It keeps the exact Pillow RGB/bilinear and binary-mask/nearest-neighbor operations, existing Protocol B labels, duplicate rules, five-view groups, and split selection. The regression fixture produces identical pixels, record manifests, labels and partitions to the archived serial implementation, including grayscale, RGBA, JPEG-in-PNG filenames, and missing/empty masks.

## Switch from the running old version

1. In the terminal running the old preparation, press **Ctrl+C** and wait for the shell prompt. The new program detects another `prepare.py` process and refuses concurrent preparation; do not run two versions together.
2. Keep the existing `prepared/` folder and PNG files. Run from this experiment folder:

```bash
bash run.sh prepare --resume --workers 4
```

3. After `Preparation COMPLETE` and `Exit status: 0`, continue with the two smoke commands in `RUN_COMMANDS.txt`.

`FAST_PREPARE_COMMANDS.txt` contains the short copy/paste commands. The default remains four workers when `--workers` is omitted. If a busy source disk performs worse with four readers, stop with Ctrl+C and resume with `--workers 2`. Worker count and PNG compression are operational choices, not changes to the scientific split or pixel recipe.

## Where the speed comes from

- **Parallel work:** at most twice the worker count is queued, bounding decoded-image memory and avoiding hundreds of thousands of queued futures.
- **One source read:** source bytes are read once, SHA-256 checked against the audit, then decoded from that same buffer. The old implementation reopened files for decoding after hashing.
- **Faster lossless PNG encoding:** new files use compression level 1 instead of the default 6. Decoded pixels and pixel-hash filenames are unchanged; encoded bytes and file checksums can differ. Valid existing PNGs retain their original encoding.
- **Reuse:** existing PNGs are fully decoded and checked against the expected pixels before adoption. They are not blindly trusted or discarded. Repeated masks are reused. Broken/incomplete cache files are atomically replaced from verified raw inputs.
- **No routine final full-cache reread:** encoded file hashes are calculated from the exact bytes written or read during reuse and carried into the checksum manifest. Add `--verify-cache` if an independent full disk reread is desired; that adds I/O. Training/evaluation still independently verify cached pixel hashes when loading data.
- **Resume:** `.prepare_journal.sqlite3` stores completed per-image results, source file identity/size/timestamps, and encoded-cache hashes. On resume, unchanged source metadata and matching cache hashes allow raw decoding to be skipped. Changed source metadata triggers a fresh audit-hash check; missing or corrupt caches are regenerated selectively. Recipe, audit, configuration, and Pillow mismatches refuse journal reuse.

The old serial program had no per-image source-to-cache journal. Its existing files are preserved, but the **first migration still has to audit/decode earlier source images once** to identify and verify those cached outputs; it skips rewriting them. Later fast-version resumes use the journal and can skip raw reads/decoding. No claim is made that the legacy run's entire elapsed work can be skipped immediately.

The parent commits journal results during processing and on orderly interruption. A small in-flight batch can finish after Ctrl+C; wait for the prompt. Such PNGs remain reusable even if their results had not reached the journal. Atomic PNG publication prevents partially visible new writes. An output-folder advisory lock protects new writers; process detection also checks the old version, which did not take that lock.

## Progress and completion

The terminal reports the phase and, during cache construction, completion percentage, records/second, ETA, journal-resumed records, decoded records, and new/reused/repaired files every five seconds. This is a live average, not a guaranteed finish time. Disk contention and image sizes vary.

- `prepared/progress.json`: operational progress/status; not a frozen manifest.
- `prepared/.prepare_journal.sqlite3`: resume journal; keep it for interruption recovery.
- `prepared/preparation_runtime.json`: final worker settings and work counters.
- `prepared/COMPLETE.json`: authoritative completion marker; partial output is never treated as ready for training.

Running preparation again on a completed, valid output exits without redoing transforms. `--verify-cache` can also validate an already completed cache. Progress, lock and SQLite files are intentionally excluded from immutable artifact hashes.

## Why this version keeps preparation on CPU

Most source files are PNGs. This path must still read/decode them, hash source bytes, and encode/write PNG outputs. A GPU resize alone leaves those costs and adds transfers; replacing Pillow resampling would also require a separately validated pixel recipe. The implemented optimization parallelizes the current native decoding/resizing/encoding and improves I/O without adding GPU dependencies or changing the comparison inputs. GPU model fitting remains in the later explicit commands.

The synthetic timing check in `logs/prepare_synthetic_benchmark.json` compares warm-cache invented RGB images in `/tmp`; it checks identical pixel hashes. It excludes full audit, mask, journal and real-drive costs, so its ratio must not be advertised as the full Real-IAD speedup. Use the live records/second after restart to assess this machine's real throughput.

The serial setup/lock is archived under `provenance/revisions/before_fast_prepare/`. The existing real-data process/cache was left untouched during implementation and testing. No new real preparation, GPU smoke check, training, or evaluation was launched automatically.
