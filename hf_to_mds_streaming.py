#!/usr/bin/env python3
"""
HF → MDS converter/uploader with streaming + progress

Key behavior for this version:
- **Single output folder per split** (no per-worker subdirs). You get one `train/` with dozens of shards.
- **Parallel readers + single writer**: N producers stream HF shards into a queue; one writer process rolls shards at the configured size.
- **End-of-run upload** to Hugging Face Hub (safer than many concurrent commits). Use `--upload-during` to push after each N records.

Usage example (Wikipedia, streaming, default shard size):
  python hf_to_mds_streaming.py \
    --repo-id wikimedia/wikipedia --config 20231101.en --split train \
    --out-local ./mds/wikipedia/20231101.en \
    --out-hub bgub/wikipedia-20231101-en-mds \
    --procs 16 --streaming --expected-num-records 6410000 --upload-after

Env:
  export HF_TOKEN=...  # for Hub writes / gated datasets

"""

import os
import sys
import time
import json
import signal
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, List

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from streaming import MDSWriter
from streaming.base.util import merge_index

# Optional imports for fallback parquet iteration
try:
    from datasets.filesystems import HfFileSystem  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional during import
    HfFileSystem = None  # type: ignore
    pq = None  # type: ignore

IS_DARWIN = sys.platform == "darwin"

# -------------------------- Column inference ---------------------------------


def _dtype_of(value: Any) -> str:
    if value is None:
        return "json"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, bool):
        return "json"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    if isinstance(value, str):
        return "str"
    return "json"


def infer_columns(ds, fallback_sample: Optional[dict] = None) -> Dict[str, str]:
    feats = getattr(ds, "features", None)
    if feats:
        cols = {}
        for name, feat in feats.items():
            t = str(getattr(feat, "dtype", ""))
            if t in ("string", "large_string"):
                cols[name] = "str"
            elif t == "binary":
                cols[name] = "bytes"
            elif t.startswith("int"):
                cols[name] = "int64"
            elif t.startswith("float"):
                cols[name] = "float64"
            else:
                cols[name] = "json"
        if cols:
            return cols
    sample = fallback_sample
    if sample is None:
        it = iter(ds)
        sample = next(it)
    return {k: _dtype_of(v) for k, v in sample.items()}


# -------------------------- Signals -----------------------------------------

_STOP = None  # set in main with proper multiprocessing context


def _install_signal_handlers(stop):
    def _handle(sig, frame):
        with stop.get_lock():
            stop.value = True

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            pass


# -------------------------- HF Hub helpers ----------------------------------


def _hf_repo_parts(remote_root: str) -> tuple[str, str]:
    """Return (repo_id, path_in_repo) from an hf:// URL.

    Accepts: hf://datasets/<org>/<name>[/<sub/dirs>]
    """
    from urllib.parse import urlparse

    if not remote_root.startswith("hf://"):
        raise ValueError(f"Expected hf:// URL, got: {remote_root}")
    parsed = urlparse(remote_root)
    path = parsed.path.lstrip("/")
    parts = path.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Malformed HF URL (need org/name): {remote_root}")
    org, name = parts[0], parts[1]
    subpath = parts[2] if len(parts) > 2 else ""
    return f"{org}/{name}", subpath


def _upload_folder_hf(
    local_dir: str, remote_root: Optional[str], api: Optional[HfApi]
) -> None:
    if not (remote_root and remote_root.startswith("hf://")):
        return
    api = api or HfApi()
    repo_id, path_in_repo = _hf_repo_parts(remote_root)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"Add/update {os.path.basename(local_dir)}",
    )


# -------------------------- Producer/Writer ---------------------------------


@dataclass
class WriterOpts:
    out_local: str
    out_remote: Optional[str]
    columns: Dict[str, str]
    compression: Optional[str]
    size_limit: Optional[str]
    upload_threads: int
    hash_alg: Optional[str]


def _make_writer(opts: WriterOpts) -> MDSWriter:
    # Build kwargs without passing None values so MDSWriter uses its own defaults
    kwargs = dict(
        out=opts.out_local,
        columns=opts.columns,
        keep_local=True,
        exist_ok=True,
        max_workers=opts.upload_threads,
        progress_bar=False,
    )
    if opts.compression is not None:
        kwargs["compression"] = opts.compression
    if opts.size_limit is not None:
        kwargs["size_limit"] = opts.size_limit  # omit to get ~64MB default
    if opts.hash_alg is not None:
        kwargs["hashes"] = [opts.hash_alg]
    return MDSWriter(**kwargs)


def _producer(
    rank: int,
    world: int,
    q: Any,
    stop: Any,
    repo_id: str,
    config: Optional[str],
    split: str,
    revision: Optional[str],
):
    """Stream examples from HF. On schema cast errors, fall back to direct Parquet iteration.

    The fallback enumerates Parquet files under data/<config>/<split>/ on the Hub and yields
    rows without casting, allowing extra columns to be ignored downstream.
    """
    try:
        ds = load_dataset(
            repo_id, config, split=split, revision=revision, streaming=True
        )
        # Guard sharding: tiny configs may not have enough sources for all ranks.
        try:
            part = ds.shard(num_shards=world, index=rank)
        except Exception:
            # If this rank would be empty, exit early; let rank 0 process the whole stream.
            if rank != 0:
                q.put(None)
                return
            part = ds

        for ex in part:
            if stop.value:
                break
            q.put(ex)
        q.put(None)  # sentinel
        return
    except Exception as e:
        # CastError from datasets when file columns don't match the dataset features.
        # We detect broadly and switch to a robust fallback.
        msg = str(e)
        cast_error = (
            "CastError" in msg
            or "Couldn't cast" in msg
            or "column names don't match" in msg
        )
        if not cast_error:
            # Unknown error; propagate to terminate this worker but keep others alive.
            q.put(None)
            return

    # Fallback: direct Parquet iteration via HfFileSystem + pyarrow
    print(
        "Note: schema mismatch while streaming; falling back to direct Parquet read.",
        flush=True,
    )
    if HfFileSystem is None or pq is None:
        # Required optional deps not available; give up cleanly.
        q.put(None)
        return

    try:
        _producer_parquet_fallback(
            rank=rank,
            world=world,
            q=q,
            stop=stop,
            repo_id=repo_id,
            config=config,
            split=split,
            revision=revision,
        )
    finally:
        q.put(None)


def _list_parquet_files(
    api: HfApi, repo_id: str, config: Optional[str], split: str, revision: Optional[str]
) -> List[str]:
    """List parquet file paths in the repo for a given config/split.

    Returns repo-relative paths like 'data/<config>/<split>/000_00000.parquet'.
    """
    # Common HF layout used by parquet exports
    search_paths: List[str] = []
    if config:
        search_paths.append(f"data/{config}/{split}/")
    # Some datasets might place split directly under data/
    search_paths.append(f"data/{split}/")

    files: List[str] = []
    for p in search_paths:
        try:
            files_in_path = api.list_repo_files(
                repo_id=repo_id, repo_type="dataset", paths=[p], revision=revision
            )
        except Exception:
            files_in_path = []
        for f in files_in_path:
            if f.endswith(".parquet") and "/_temporary/" not in f:
                files.append(f)
    # Stable ordering for deterministic sharding
    files.sort()
    return files


def _producer_parquet_fallback(
    rank: int,
    world: int,
    q: Any,
    stop: Any,
    repo_id: str,
    config: Optional[str],
    split: str,
    revision: Optional[str],
    read_batch_size: int = 4096,
) -> None:
    """Fallback producer that reads parquet files directly to avoid schema casts.

    It shards the list of files across workers (by index mod world) and streams rows.
    Only columns required by the consumer will be kept downstream, so extra columns are fine.
    """
    api = HfApi()
    files = _list_parquet_files(api, repo_id, config, split, revision)
    if not files:
        return

    # Build base hf:// path with optional revision pin
    base = f"hf://datasets/{repo_id}"
    if revision:
        base = f"{base}@{revision}"

    fs = HfFileSystem()

    for idx, rel_path in enumerate(files):
        if stop.value:
            break
        if (idx % world) != rank:
            continue
        hf_path = f"{base}/{rel_path}"
        try:
            with fs.open(hf_path, "rb") as fobj:
                parquet = pq.ParquetFile(fobj)
                num_row_groups = parquet.num_row_groups
                for rg in range(num_row_groups):
                    if stop.value:
                        break
                    table = parquet.read_row_group(rg)
                    # Stream rows in manageable chunks to avoid large memory spikes
                    num_rows = table.num_rows
                    start = 0
                    while start < num_rows and not stop.value:
                        end = min(start + read_batch_size, num_rows)
                        batch = table.slice(start, end - start)
                        for row in batch.to_pylist():
                            if stop.value:
                                break
                            # Emit raw row; consumer will select expected columns
                            q.put(row)
                        start = end
        except Exception:
            # Skip unreadable file and continue
            continue


def _consumer_writer(
    q: Any,
    stop: Any,
    opts: WriterOpts,
    expected_total: Optional[int] = None,
    upload_during: bool = False,
    upload_every: int = 250000,
    remote_api: Optional[HfApi] = None,
):
    written = 0
    pbar = tqdm(total=expected_total, unit="ex", desc="write", leave=True)
    with _make_writer(opts) as w:
        alive_producers = 0
        # Count producers via special start message
        # We’ll detect completion by counting None sentinels
        while True:
            item = q.get()
            if item is None:
                alive_producers += 1
                # when we've seen N sentinels, we're done. N injected by main.
                if alive_producers >= _SENTINELS_EXPECTED:
                    break
                continue
            if stop.value:
                break
            row = {k: item.get(k) for k in opts.columns.keys()}
            w.write(row)
            written += 1
            pbar.update(1)
            if upload_during and (written % upload_every == 0) and opts.out_remote:
                _upload_folder_hf(opts.out_local, opts.out_remote, remote_api)
    pbar.close()


def _effective_world(ds, requested: int) -> int:
    """Return how many producer ranks have non-empty sources for sharding.

    Tries ds.shard(num_shards=requested, index=i) for each rank and counts successes.
    Falls back to 1 if none succeed.
    """
    ok = 0
    for i in range(requested):
        try:
            ds.shard(num_shards=requested, index=i)
            ok += 1
        except Exception:
            pass
    return max(1, ok or 1)


# Global shared count of sentinels expected (set in main)
_SENTINELS_EXPECTED = 0

# ------------------------------ Main ----------------------------------------


def main():
    global _SENTINELS_EXPECTED
    global _SENTINELS_EXPECTED, _STOP
    ctx = mp.get_context("spawn") if IS_DARWIN else mp
    _STOP = ctx.Value("b", False)
    _install_signal_handlers(_STOP)

    ap = argparse.ArgumentParser(
        description="HF split → single-folder MDS (parallel streaming readers + single writer)"
    )
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None)

    ap.add_argument("--out-local", required=True)
    ap.add_argument("--out-hub", default=None)
    ap.add_argument(
        "--dest-subdir",
        default=None,
        help="Subfolder under the dest repo to store this language/config (default: --config)",
    )

    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.set_defaults(streaming=True)

    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    ap.add_argument(
        "--compression",
        default="zstd",
        help="Compression codec for shards (e.g., zstd or zstd:11). Use 'none' to disable.",
    )
    ap.add_argument(
        "--size-limit",
        default=None,
        help="Shard size limit like '67mb'. Omit to keep ~64MB default",
    )
    ap.add_argument("--upload-threads", type=int, default=16)
    ap.add_argument(
        "--hash",
        default="xxh64",
        help="Shard hash algorithm (e.g., xxh64, sha1). Use 'none' to disable.",
    )

    ap.add_argument("--readme", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--make-public", action="store_true")

    ap.add_argument("--expected-num-records", type=int, default=None)
    ap.add_argument(
        "--upload-after",
        action="store_true",
        help="Upload to Hub only once after merge",
    )
    ap.add_argument(
        "--upload-during",
        action="store_true",
        help="Also upload mid-run (every ~250k records)",
    )

    args = ap.parse_args()

    os.makedirs(args.out_local, exist_ok=True)
    dest_subdir = args.dest_subdir or (args.config or "default")
    local_split_root = os.path.join(args.out_local, dest_subdir, args.split)
    os.makedirs(local_split_root, exist_ok=True)

    remote_split_root = None
    api = None
    if args.out_hub:
        remote_split_root = (
            f"hf://datasets/{args.out_hub}/{dest_subdir}/{args.split}".rstrip("/")
        )
        api = HfApi()
        api.create_repo(
            repo_id=args.out_hub,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    # Probe for columns using streaming handle
    probe = load_dataset(
        args.repo_id,
        args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    sample = None
    try:
        if not getattr(probe, "features", None):
            sample = next(iter(probe))
    except StopIteration:
        print("Split appears empty.")
        return 0
    columns = infer_columns(probe, fallback_sample=sample)
    print("Columns:")
    for k, v in columns.items():
        print(f"  - {k}: {v}")

    # Build queue and start producers
    world_req = max(1, int(args.procs))
    # Reduce parallelism for tiny configs where some ranks would be empty.
    world = _effective_world(probe, world_req)
    if world < world_req:
        print(f"Note: reducing producers from {world_req} → {world} for this split.")
    q: mp.Queue = ctx.Queue(maxsize=4096)
    procs = []
    _SENTINELS_EXPECTED = world
    for rank in range(world):
        p = ctx.Process(
            target=_producer,
            args=(
                rank,
                world,
                q,
                _STOP,
                args.repo_id,
                args.config,
                args.split,
                args.revision,
            ),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # Writer (single folder)
    writer_opts = WriterOpts(
        out_local=local_split_root,
        out_remote=remote_split_root,
        columns=columns,
        compression=(
            None
            if (args.compression is None or str(args.compression).lower() == "none")
            else args.compression
        ),
        size_limit=args.size_limit,
        upload_threads=args.upload_threads,
        hash_alg=(
            None
            if (args.hash is None or str(args.hash).lower() == "none")
            else args.hash
        ),
    )
    _consumer_writer(
        q,
        _STOP,
        writer_opts,
        expected_total=args.expected_num_records,
        upload_during=args.upload_during,
        remote_api=api,
    )

    # Join producers
    for p in procs:
        p.join()

    # Merge index for the single-folder dataset
    print("Merging index…")
    merge_index(local_split_root, keep_local=True)

    # Upload final contents
    if remote_split_root and (args.upload_after or args.out_hub):
        _upload_folder_hf(local_split_root, remote_split_root, api)

    # README & visibility
    if api and args.readme and os.path.exists(args.readme):
        api.upload_file(
            path_or_fileobj=args.readme,
            path_in_repo="README.md",
            repo_id=args.out_hub,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    if api and args.make_public:
        api.update_repo_visibility(
            repo_id=args.out_hub, repo_type="dataset", private=False
        )

    print("Done.")
    print(f"Local split root: {local_split_root}")
    if remote_split_root:
        print(f"Uploaded to: {remote_split_root}")


if __name__ == "__main__":
    if IS_DARWIN:
        mp.set_start_method("spawn", force=True)
    sys.exit(main())
