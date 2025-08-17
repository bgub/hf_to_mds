#!/usr/bin/env python3
# pip install datasets huggingface_hub
import os, re, shutil, subprocess, sys, time
from argparse import ArgumentParser
from typing import List
from datasets import get_dataset_config_names, get_dataset_split_names
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
        "--procs", type=int, default=16, help="# workers passed to converter"
    )
    ap.add_argument(
        "--hash", default="xxh64", help="xxh64 | sha1 | none (passed to converter)"
    )
    ap.add_argument(
        "--size-limit", default=None, help="e.g., 67mb, 256mb (omit for default)"
    )
    ap.add_argument(
        "--compression", default=None, help="e.g., zstd or zstd:11 (omit for none)"
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

    # Helper to build converter cmd
    def converter_cmd(cfg: str, split: str) -> List[str]:
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
            cfg,  # mirror config name
            "--procs",
            str(args.procs),
            "--streaming",
            "--upload-after",
            "--hash",
            args.hash,
        ]
        if args.size_limit:
            cmd += ["--size-limit", args.size_limit]
        if args.compression:
            cmd += ["--compression", args.compression]
        return cmd

    total_jobs = 0
    for cfg in cfgs:
        # Enumerate splits for this config
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
            total_jobs += 1
            print(f"\n=== {args.src} | {cfg} | {split} ===")
            if not args.force and hub_has_index(api, args.out_hub, cfg, split):
                print("✓ Skip — already on Hub.")
                continue

            cmd = converter_cmd(cfg, split)
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
                continue

            rc = run(cmd)
            if rc != 0:
                print(f"✗ Failed ({rc}). Keeping local files for inspection.")
                continue

            # Verify upload then delete local artifacts for this cfg/split
            # if hub_has_index(api, args.out_hub, cfg, split):
            if True:
                if not args.no_delete:
                    local_dir = os.path.join(
                        args.out_local,
                        cfg,
                        split.rsplit("/", 1)[0] if "/" in split else "",
                        split,
                    )
                    # More robust: nuke the whole config directory for this run
                    local_cfg_dir = os.path.join(args.out_local, cfg)
                    target = (
                        local_cfg_dir if os.path.isdir(local_cfg_dir) else local_dir
                    )
                    try:
                        shutil.rmtree(target)
                        print(f"🧹 Deleted local {target}")
                    except Exception as e:
                        print(f"Warning: could not delete {target}: {e}")
            else:
                print("Warning: index.json not found on Hub; not deleting local copy.")

            time.sleep(args.sleep)

    print(f"\nDone. Processed {total_jobs} job(s).")


if __name__ == "__main__":
    main()
