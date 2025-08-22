# HuggingFace to MDS Converter

Convert HuggingFace datasets to [MosaicML Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/index.html) format (MDS) for efficient cloud-based training.

## Installation

```bash
pip install datasets huggingface_hub mosaicml-streaming
```

## Quick Start

**Batch convert entire dataset:**

```bash
python batch_to_mds.py \
  --src wikimedia/wikipedia \
  --out-hub bgub/wikipedia-mds-test \
  --out-local ./mds-local-2/wikipedia \
  --procs 10
```

**Convert single config/split:**

```bash
python hf_to_mds_streaming.py \
  --repo-id HuggingFaceFW/fineweb \
  --split train \
  --out-local /mnt/mds/fineweb \
  --out-hub ben-gubler/fineweb-mds \
  --procs 16 \
  --streaming
```

## Key Options

**`batch_to_mds.py`** - Batch convert all configs/splits:

- `--src` / `--out-hub`: Source and destination repos (required)
- `--procs`: Total worker processes across all subsets; jobs run concurrently (default: 16)
- `--records-per-proc`: Approximate records per worker when sizing each job (default: 1,000,000)
- `--compression`: e.g., `zstd`, `zstd:11`
- `--include-config` / `--exclude-config`: Regex filters
- `--dry-run`: Preview without executing
- `--force`: Rebuild existing datasets

**`hf_to_mds_streaming.py`** - Single config/split converter (called by batch script)

## Examples

```bash
# Convert specific language only
python batch_to_mds.py \
  --src wikimedia/wikipedia \
  --out-hub your-username/wikipedia-en-mds \
  --include-config "^20231101\.en$"

# Preview what would be processed
python batch_to_mds.py \
  --src microsoft/orca-math-word-problems-200k \
  --out-hub your-username/orca-math-mds \
  --dry-run
```

## Using Converted Datasets

```python
from streaming import StreamingDataset
from torch.utils.data import DataLoader

dataset = StreamingDataset(remote='hf://your-username/dataset-mds')
dataloader = DataLoader(dataset, batch_size=32)
```

## Benefits

MDS format provides:

- **Elastic Determinism**: Reproducible across hardware configs
- **Fast Resumption**: Resume training in seconds
- **High Throughput**: Optimized for cloud streaming
- **Effective Shuffling**: Maintains quality while reducing costs
