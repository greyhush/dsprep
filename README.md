# dsprep

Training dataset preparation toolkit. Convert formats, deduplicate, filter by quality, get stats.

No dependencies. CLI-first. Works with any fine-tuning framework.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Auto-detect format
dsprep detect data/train.jsonl

# Convert ShareGPT -> ChatML (the format Qwen/LLaMA expect)
dsprep convert sharegpt_data.jsonl chatml_data.jsonl --format chatml

# Convert Alpaca -> ShareGPT
dsprep convert alpaca_data.json sharegpt_data.jsonl --format sharegpt

# Deduplicate
dsprep dedup data.jsonl --output clean.jsonl

# Filter: remove too-short entries, require non-empty output field
dsprep filter data.jsonl --min-length 50 --required-fields output -o filtered.jsonl

# Filter: Chinese-only
dsprep filter data.jsonl --language zh -o zh_only.jsonl

# Stats overview
dsprep stats data.jsonl

# Stats with conversation-level analysis
dsprep stats data.jsonl --conversations

# Random samples
dsprep sample data.jsonl -n 5

# Preview first N entries
dsprep head data.jsonl -n 10

# Merge multiple files
dsprep merge part1.jsonl part2.jsonl part3.jsonl -o merged.jsonl

# Split into train/val/test
dsprep split data.jsonl -o split/ --test-ratio 0.1 --val-ratio 0.1
```

## Supported Formats

| Format | Description | Example |
|--------|-------------|---------|
| `alpaca` | Stanford Alpaca | `{"instruction": "...", "input": "...", "output": "..."}` |
| `sharegpt` | ShareGPT | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` |
| `chatml` | ChatML (Qwen/LLaMA) | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| `completion` | Raw text | `{"text": "..."}` |
| `jsonl` | Generic JSONL | Auto-detects fields |

## Python API

```python
from dsprep import detect_format, convert, deduplicate, filter_dataset, dataset_stats

# Auto-detect format
fmt = detect_format("data.jsonl")

# Convert
convert("sharegpt.jsonl", "chatml.jsonl", target_fmt="chatml")

# Load and deduplicate
from dsprep.formats import load
entries = load("data.jsonl")
clean, dupes = deduplicate(entries)

# Filter
filtered, stats = filter_dataset(entries, min_length=50, language="zh")

# Stats
stats = dataset_stats(entries)
```

## Commands

| Command | Description |
|---------|-------------|
| `dsprep detect FILE` | Auto-detect format |
| `dsprep convert IN OUT` | Convert format |
| `dsprep dedup FILE` | Remove duplicates |
| `dsprep filter FILE` | Quality filter |
| `dsprep stats FILE` | Statistics |
| `dsprep sample FILE` | Random samples |
| `dsprep head FILE` | First N entries |
| `dsprep merge FILES` | Merge files |
| `dsprep split FILE` | Train/val/test split |

## License

MIT
