#!/usr/bin/env python3
# pip install datasets huggingface_hub
import os
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

from datasets import (
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_split_names,
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


def run(cmd: List[str]) -> int:
    print(" ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    return p.returncode


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
        "--max-workers", type=int, default=16, help="Total workers across all jobs"
    )
    ap.add_argument(
        "--max-workers-per-job", type=int, default=8, help="Worker cap per job"
    )
    ap.add_argument(
        "--records-per-worker",
        type=int,
        default=500_000,
        help="Approximate records handled by each worker",
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

    # Enumerate configs (subsets)
    cfgs = get_dataset_config_names(args.src)
    inc_cfg, exc_cfg = re.compile(args.include_config), re.compile(args.exclude_config)
    cfgs = [c for c in cfgs if inc_cfg.search(str(c)) and not exc_cfg.search(str(c))]
    print(f"Found {len(cfgs)} configs after filtering.")

    @dataclass
    class Job:
        cfg: str
        split: str
        procs: int

    def estimate_workers(cfg: str, split: str) -> int:
        try:
            info = get_dataset_config_info(args.src, cfg)
            n = int(info.splits[split].num_examples)  # type: ignore
        except Exception:
            n = 0
        if n <= 0:
            return 1
        need = n // args.records_per_worker + 1
        return min(args.max_workers_per_job, max(1, need))

    # Helper to build converter cmd
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

    jobs: Deque[Job] = deque()

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

        for split in splits:
            print(f"\n=== {args.src} | {cfg} | {split} ===")
            if not args.force and hub_has_index(api, args.out_hub, cfg, split):
                print("✓ Skip — already on Hub.")
                continue
            procs = estimate_workers(cfg, split)
            jobs.append(Job(cfg, split, procs))

    if args.dry_run:
        for job in jobs:
            cmd = converter_cmd(job.cfg, job.split, job.procs)
            print("DRY RUN:", " ".join(cmd))
        print(f"\nDone. Processed {len(jobs)} job(s).")
        return

    running: List[Tuple[subprocess.Popen, Job]] = []
    available = args.max_workers
    total_jobs = len(jobs)

    while jobs or running:
        while jobs and jobs[0].procs <= available:
            job = jobs.popleft()
            cmd = converter_cmd(job.cfg, job.split, job.procs)
            print(" ".join(cmd), flush=True)
            p = subprocess.Popen(cmd)
            running.append((p, job))
            available -= job.procs

        time.sleep(args.sleep)

        for proc, job in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            running.remove((proc, job))
            available += job.procs
            if rc != 0:
                print(
                    f"✗ {job.cfg}/{job.split} failed ({rc}). Keeping local files for inspection."
                )
                continue
            if not args.no_delete:
                local_dir = os.path.join(
                    args.out_local,
                    job.cfg,
                    job.split.rsplit("/", 1)[0] if "/" in job.split else "",
                    job.split,
                )
                local_cfg_dir = os.path.join(args.out_local, job.cfg)
                target = local_cfg_dir if os.path.isdir(local_cfg_dir) else local_dir
                try:
                    shutil.rmtree(target)
                    print(f"🧹 Deleted local {target}")
                except Exception as e:
                    print(f"Warning: could not delete {target}: {e}")

    print(f"\nDone. Processed {total_jobs} job(s).")


if __name__ == "__main__":
    main()
