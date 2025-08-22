#!/usr/bin/env python3
# pip install datasets huggingface_hub
import os, re, shutil, sys, asyncio
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    get_dataset_config_info,
)
from huggingface_hub import HfApi


def hub_has_index(api: HfApi, repo_id: str, cfg: str, split: str) -> bool:
    # Looks for the merged index.json we upload to <cfg>/<split>/index.json
    try:
        path = f"{cfg}/{split}/index.json"
        files = api.list_repo_files(
            repo_id=repo_id, repo_type="dataset", paths=[f"{cfg}/{split}/"]
        )
        return path in files
    except Exception:
        return False


@dataclass
class Job:
    cfg: str
    split: str
    procs: int
    size: Optional[int]


class ProcLimiter:
    def __init__(self, total: int):
        self.total = total
        self._avail = total
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def limit(self, n: int):
        async with self._cond:
            await self._cond.wait_for(lambda: self._avail >= n)
            self._avail -= n
        try:
            yield
        finally:
            async with self._cond:
                self._avail += n
                self._cond.notify_all()


def main():
    ap = ArgumentParser(
        description="Batch convert ANY HF dataset to MDS across all configs & splits"
    )
    ap.add_argument(
        "--src", required=True, help="Source dataset repo (e.g., wikimedia/wikipedia)"
    )
    ap.add_argument(
        "--out-hub",
        required=True,
        help="Destination HF dataset repo (e.g., bgub/wikipedia-mds)",
    )
    ap.add_argument("--out-local", default="./mds_batch", help="Local staging root")
    ap.add_argument(
        "--converter",
        default="hf_to_mds_streaming.py",
        help="Path to the converter script you put in the canvas",
    )
    ap.add_argument(
        "--procs", type=int, default=16, help="# workers for largest subsets"
    )
    ap.add_argument(
        "--total-procs",
        type=int,
        default=None,
        help="Total worker processes to share across jobs (default: --procs)",
    )
    ap.add_argument(
        "--hash", default="xxh64", help="xxh64 | sha1 | none (passed to converter)"
    )
    ap.add_argument(
        "--size-limit", default=None, help="e.g., 67mb, 256mb (omit for default)"
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        help="Compression for shards (e.g., zstd or zstd:11). Use 'none' to disable.",
    )
    ap.add_argument(
        "--include-config", default=r".*", help="Regex filter for configs (subsets)"
    )
    ap.add_argument("--exclude-config", default=r"$^", help="Regex to exclude configs")
    ap.add_argument(
        "--include-split", default=r".*", help="Regex filter for splits (train|test|…)"
    )
    ap.add_argument("--exclude-split", default=r"$^", help="Regex to exclude splits")
    ap.add_argument(
        "--force", action="store_true", help="Rebuild even if index exists on Hub"
    )
    ap.add_argument(
        "--no-delete",
        action="store_true",
        help="Do NOT delete local files after upload",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--sleep", type=float, default=1.0, help="Pause between jobs (seconds)"
    )
    args = ap.parse_args()

    api = HfApi()
    os.makedirs(args.out_local, exist_ok=True)

    if args.total_procs is None:
        args.total_procs = args.procs

    # Enumerate configs (subsets)
    cfgs = get_dataset_config_names(args.src)
    inc_cfg, exc_cfg = re.compile(args.include_config), re.compile(args.exclude_config)
    cfgs = [c for c in cfgs if inc_cfg.search(str(c)) and not exc_cfg.search(str(c))]
    print(f"Found {len(cfgs)} configs after filtering.")

    def determine_procs(num_examples: Optional[int]) -> int:
        if num_examples is None:
            return 1
        if num_examples < 1_000_000:
            return 1
        if num_examples < 5_000_000:
            return min(2, args.procs)
        if num_examples < 20_000_000:
            return min(4, args.procs)
        return args.procs

    def converter_cmd(cfg: str, split: str, procs: int) -> List[str]:
        cmd = [
            sys.executable,
            args.converter,
            "--repo-id",
            args.src,
            "--config",
            cfg,
            "--split",
            split,
            "--out-local",
            args.out_local,
            "--out-hub",
            args.out_hub,
            "--dest-subdir",
            cfg,
            "--procs",
            str(procs),
            "--streaming",
            "--upload-after",
            "--hash",
            args.hash,
        ]
        if args.size_limit:
            cmd += ["--size-limit", args.size_limit]
        if args.compression:
            cmd += [
                "--compression",
                ("none" if str(args.compression).lower() == "none" else args.compression),
            ]
        return cmd

    jobs: List[Job] = []
    pending_cfg_counts: Dict[str, int] = {}

    for cfg in cfgs:
        try:
            splits = get_dataset_split_names(args.src, cfg)
        except Exception as e:
            print(f"Warning: could not list splits for {cfg}: {e}; assuming ['train']")
            splits = ["train"]

        inc_sp, exc_sp = re.compile(args.include_split), re.compile(args.exclude_split)
        splits = [s for s in splits if inc_sp.search(s) and not exc_sp.search(s)]
        if not splits:
            continue
        pending_cfg_counts[cfg] = len(splits)

        info = None
        try:
            info = get_dataset_config_info(args.src, cfg)
        except Exception:
            pass

        for split in splits:
            size = None
            if info and info.splits and split in info.splits:
                size = info.splits[split].num_examples
            procs = determine_procs(size)
            jobs.append(Job(cfg, split, procs, size))

    limiter = ProcLimiter(args.total_procs)

    async def run_job(job: Job):
        cfg, split, procs, size = job.cfg, job.split, job.procs, job.size
        print(f"\n=== {args.src} | {cfg} | {split} (procs={procs}) ===")
        if not args.force and hub_has_index(api, args.out_hub, cfg, split):
            print("✓ Skip — already on Hub.")
            return
        cmd = converter_cmd(cfg, split, procs)
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            return
        async with limiter.limit(procs):
            print(" ".join(cmd), flush=True)
            proc = await asyncio.create_subprocess_exec(*cmd)
            rc = await proc.wait()
        if rc != 0:
            print(f"✗ Failed ({rc}). Keeping local files for inspection.")
            return
        if not args.no_delete and hub_has_index(api, args.out_hub, cfg, split):
            pending_cfg_counts[cfg] -= 1
            if pending_cfg_counts[cfg] == 0:
                local_cfg_dir = os.path.join(args.out_local, cfg)
                try:
                    shutil.rmtree(local_cfg_dir)
                    print(f"🧹 Deleted local {local_cfg_dir}")
                except Exception as e:
                    print(f"Warning: could not delete {local_cfg_dir}: {e}")
        await asyncio.sleep(args.sleep)

    async def runner():
        await asyncio.gather(*(run_job(j) for j in jobs))

    asyncio.run(runner())

    print(f"\nDone. Processed {len(jobs)} job(s).")


if __name__ == "__main__":
    main()
