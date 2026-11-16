"""Parallel, lossless cache construction. No Torch/CUDA or model dependencies."""
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from common import now, sha256, write_json

# Change this if pixel transforms, mask semantics, or stored result semantics change.
RECIPE_VERSION = 'pillow-rgb-bilinear-mask-nearest-protocol-b-v1'


def other_prepare_pids(script, proc_root=Path('/proc')):
    """Catch the legacy process too: it predates our advisory output lock."""
    script = Path(script).resolve()
    active = []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            for raw in argv:
                argument = Path(os.fsdecode(raw))
                if argument.name != script.name:
                    continue
                candidate = argument if argument.is_absolute() else (proc / 'cwd').resolve() / argument
                if candidate.resolve() == script:
                    active.append(int(proc.name))
                    break
        except (OSError, ValueError):
            continue  # Unrelated processes can exit or be owned by another user.
    return sorted(active)


def source_stamps(record, cfg):
    names = [record['raw_image']]
    if record['raw_mask'] and not record['annotation_issues']:
        names.append(record['raw_mask'])
    result = {}
    for name in names:
        stat = (Path(cfg['raw_root']) / name).stat()
        result[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


def audited_bytes(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError('Raw input differs from audited bytes: ' + str(path))
    return data


def cache_png(output, name, array, pixel_hash, compression):
    """Adopt valid legacy files; repair interrupted files; atomically publish new PNGs."""
    import numpy as np
    from PIL import Image
    path = Path(output) / name
    state = 'written'
    if path.exists():
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                same = (image.format == 'PNG' and np.asarray(image).shape == array.shape
                        and hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == pixel_hash)
            if same:
                return dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)), 'reused'
        except (OSError, ValueError):
            pass  # An interrupted legacy writer can leave a partial PNG.
        state = 'repaired'
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG', compress_level=compression)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.png_', suffix='.tmp', dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        # Same content-addressed output has identical pixels/encoding across these workers.
        # Atomic replacement means readers never see a partially written PNG.
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)), state


def valid_saved_result(saved, record, cfg, output):
    if saved['source_stamps'] != source_stamps(record, cfg):
        return False
    for entry in saved['files']:
        path = Path(output) / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            return False
    return True


def process_record(record, expected, cfg, output, compression=1, saved=None):
    """Decode each audited source once, keeping original Pillow pixel operations."""
    import numpy as np
    from PIL import Image
    if saved is not None and valid_saved_result(saved, record, cfg, output):
        return saved, dict(resumed=1, decoded=0, written=0, reused=len(saved['files']), repaired=0)
    stamps = source_stamps(record, cfg)
    resampling = getattr(Image, 'Resampling', Image)
    raw = Path(cfg['raw_root'])
    files, stats = [], dict(resumed=0, decoded=1, written=0, reused=0, repaired=0)

    def save(array, name, pixel_hash):
        entry, state = cache_png(output, name, array, pixel_hash, compression)
        files.append(entry)
        stats[state] += 1

    data = audited_bytes(raw / record['raw_image'], expected['image_sha256'])
    with Image.open(io.BytesIO(data)) as source:
        rgb = source.convert('RGB').resize((cfg['image_size'], cfg['image_size']), resampling.BILINEAR)
    del data
    array = np.asarray(rgb)
    content = hashlib.sha256(array.tobytes()).hexdigest()
    image_name = 'cache/' + content + '.png'
    save(array, image_name, content)
    good_mask = bool(record['raw_mask']) and not record['annotation_issues']
    nonempty = record['mask_foreground_raw'] > 0
    updates = dict(input_pixel_sha256=content, cache_image=image_name,
        visible_eligible=record['object_label'] == 0 or (good_mask and nonempty),
        pixel_eligible=record['object_label'] == 0 or good_mask or record['is_ng_without_mask'],
        support_eligible=False, annotation_issues=list(record['annotation_issues']))
    updates['visible_label'] = record['object_label'] if updates['visible_eligible'] else None
    sizes = ((cfg['image_size'], 'cache_train_mask'), (cfg['map_size'], 'cache_eval_mask'))
    if record['is_ng_without_mask']:
        for size, key in sizes:
            binary = np.zeros((size, size), dtype=np.uint8)
            relative = Path(record['id'])
            name = str(Path('cache/empty_masks') / str(size) / relative.parent / (relative.stem + '_mask_EMPTY.png'))
            pixel_hash = hashlib.sha256(binary.tobytes()).hexdigest()
            save(binary, name, pixel_hash)
            updates[key], updates[key + '_pixel_sha256'] = name, pixel_hash
        updates['synthetic_empty_mask'] = True
    if good_mask:
        data = audited_bytes(raw / record['raw_mask'], expected['mask_sha256'])
        with Image.open(io.BytesIO(data)) as source:
            if source.size != tuple(expected['image_shape'][1::-1]):
                raise RuntimeError('Unflagged source alignment mismatch: ' + record['id'])
            mask = source.convert('L')
            for size, key in sizes:
                reduced = mask.resize((size, size), resampling.NEAREST)
                binary = (np.asarray(reduced) > 0).astype(np.uint8) * 255
                pixel_hash = hashlib.sha256(binary.tobytes()).hexdigest()
                name = 'cache/mask_%s_%d.png' % (pixel_hash, size)
                save(binary, name, pixel_hash)
                updates[key], updates[key + '_pixel_sha256'] = name, pixel_hash
                if size == cfg['image_size']:
                    updates['support_eligible'] = bool(binary.any()) and not record['protected_test_group']
                elif nonempty and not binary.any():
                    updates['pixel_eligible'] = False
                    updates['annotation_issues'].append('foreground_lost_at_metric_resolution')
    if source_stamps(record, cfg) != stamps:
        raise RuntimeError('Source file changed during preparation: ' + record['id'])
    return dict(updates=updates, source_stamps=stamps, files=files), stats


class Journal:
    """Only the parent writes SQLite; commits retain completed records on Ctrl+C."""
    def __init__(self, output, cfg, resume):
        import PIL
        path = Path(output) / '.prepare_journal.sqlite3'
        if path.exists() and not resume:
            raise RuntimeError('Preparation journal exists; use --resume')
        recipe = dict(version=RECIPE_VERSION, config=cfg, pillow=PIL.__version__)
        for name in ('manifest_inventory.csv', 'file_checks.jsonl', 'pixel_issues.csv',
                     'blank_or_nearly_uniform_images.csv', 'dimension_mismatches.csv'):
            recipe[name] = sha256(Path(cfg['audit_root']) / name)
        fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        self.connection = sqlite3.connect(str(path))
        self.connection.execute('PRAGMA synchronous=NORMAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, result TEXT NOT NULL)')
        existing = self.connection.execute("SELECT value FROM metadata WHERE key='recipe'").fetchone()
        if existing is not None and existing[0] != fingerprint:
            self.connection.close()
            raise RuntimeError('Resume recipe/audit/config/Pillow differs; preserve the journal and use a separate preparation folder')
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('recipe', ?)", (fingerprint,))
        self.connection.commit()

    def get(self, identity):
        entry = self.connection.execute('SELECT result FROM records WHERE id=?', (identity,)).fetchone()
        return json.loads(entry[0]) if entry else None

    def put(self, identity, result):
        self.connection.execute('INSERT OR REPLACE INTO records VALUES (?, ?)',
                                (identity, json.dumps(result, sort_keys=True)))

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.commit()
        self.connection.close()


def build_cache(records, checks, cfg, output, workers=4, resume=False, compression=1, progress_seconds=5):
    output = Path(output)
    (output / 'cache').mkdir(exist_ok=True)
    indices = []
    for i, record in enumerate(records):
        if record['exclude_reasons'] and not record['protected_test_group']:
            record.update(visible_eligible=False, pixel_eligible=False, support_eligible=False)
        else:
            indices.append(i)
    print('phase=cache | eligible=%d | excluded=%d | workers=%d | png_level=%d | resume=%s' %
          (len(indices), len(records) - len(indices), workers, compression, resume), flush=True)
    journal = Journal(output, cfg, resume)
    started, last_print, last_commit = time.monotonic(), 0., time.monotonic()
    completed, totals, checksums = 0, dict(resumed=0, decoded=0, written=0, reused=0, repaired=0), {}
    jobs = iter(indices)
    pending = {}

    def report(status='running', force=False):
        nonlocal last_print
        current = time.monotonic()
        if not force and current - last_print < progress_seconds:
            return
        elapsed = current - started
        rate = completed / elapsed if elapsed else 0.
        remaining = (len(indices) - completed) / rate if rate else None
        data = dict(status=status, phase='cache', timestamp=now(), completed=completed, total=len(indices),
                    workers=workers, elapsed_seconds=elapsed, records_per_second=rate,
                    eta_seconds=remaining, **totals)
        write_json(output / 'progress.json', data)
        print('cache %d/%d (%.2f%%) | %.2f records/s | ETA %s | resumed=%d decoded=%d | '
              'files new=%d reused=%d repaired=%d' % (completed, len(indices),
              100 * completed / max(1, len(indices)), rate,
              'calculating' if remaining is None else '%.1f min' % (remaining / 60),
              totals['resumed'], totals['decoded'], totals['written'], totals['reused'], totals['repaired']), flush=True)
        last_print = current

    def submit(pool):
        while len(pending) < workers * 2:
            try:
                index = next(jobs)
            except StopIteration:
                return
            record = records[index]
            expected = checks[record['raw_image']]
            future = pool.submit(process_record, record, expected, cfg, output, compression,
                                 journal.get(record['id']) if resume else None)
            pending[future] = index

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            submit(pool)
            report(force=True)
            try:
                while pending:
                    ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in ready:
                        index = pending.pop(future)
                        result, stats = future.result()
                        records[index].update(result['updates'])
                        journal.put(records[index]['id'], result)
                        for entry in result['files']:
                            old = checksums.setdefault(entry['path'], entry['sha256'])
                            if old != entry['sha256']:
                                raise RuntimeError('Conflicting cache writers/encodings; stop every other preparation process: ' + entry['path'])
                        for key in totals:
                            totals[key] += stats[key]
                        completed += 1
                    if time.monotonic() - last_commit >= 1:
                        journal.commit()
                        last_commit = time.monotonic()
                    submit(pool)
                    report()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        report(status='cache_complete', force=True)
    except BaseException:
        report(status='interrupted_or_failed', force=True)
        raise
    finally:
        journal.close()
    runtime = dict(workers=workers, png_compression=compression, cache_seconds=time.monotonic() - started,
                   eligible_records=len(indices), **totals)
    return checksums, runtime


def verify_cache(output, checksums, progress_seconds=5):
    """Optional independent disk reread; PNG hashes are already computed while writing."""
    started = last = time.monotonic()
    print('phase=verify_cache | files=%d' % len(checksums), flush=True)
    for i, (name, expected) in enumerate(sorted(checksums.items()), 1):
        if sha256(Path(output) / name) != expected:
            raise RuntimeError('Cache checksum mismatch: ' + name)
        current = time.monotonic()
        if current - last >= progress_seconds or i == len(checksums):
            print('verify_cache %d/%d | %.1fs elapsed' % (i, len(checksums), current - started), flush=True)
            last = current
