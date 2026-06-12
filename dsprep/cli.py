#!/usr/bin/env python3
"""dsprep CLI - Training dataset preparation toolkit."""

import argparse
import json
import sys
from pathlib import Path


def cmd_convert(args):
    """Convert between dataset formats."""
    from dsprep.formats import convert, detect_format

    source_fmt = args.source_format
    if not source_fmt:
        source_fmt = detect_format(args.input)
        print(f"Auto-detected format: {source_fmt}")

    count = convert(args.input, args.output, source_fmt, args.format)
    print(f"Converted {count} entries: {source_fmt} -> {args.format}")
    print(f"Output: {args.output}")


def cmd_detect(args):
    """Detect the format of a dataset file."""
    from dsprep.formats import detect_format, load
    fmt = detect_format(args.file)
    entries = load(args.file, fmt)
    print(f"Format: {fmt}")
    print(f"Entries: {len(entries)}")
    if entries:
        print(f"Sample: {json.dumps(entries[0], ensure_ascii=False)[:200]}")


def cmd_dedup(args):
    """Remove duplicate entries."""
    from dsprep.formats import load, _save_jsonl
    from dsprep.dedup import deduplicate

    entries = load(args.file, args.format)
    print(f"Loaded {len(entries)} entries")

    deduped, dupes = deduplicate(entries, key=args.key, keep=args.keep)
    print(f"Removed {dupes} duplicates")
    print(f"Remaining: {len(deduped)}")

    if args.output:
        _save_jsonl(deduped, Path(args.output))
        print(f"Saved to: {args.output}")


def cmd_filter(args):
    """Filter dataset by quality criteria."""
    from dsprep.formats import load, _save_jsonl
    from dsprep.filters import filter_dataset

    entries = load(args.file, args.format)
    print(f"Loaded {len(entries)} entries")

    required = [f.strip() for f in args.required_fields.split(",")] if args.required_fields else None

    filtered, stats = filter_dataset(
        entries,
        min_length=args.min_length,
        max_length=args.max_length,
        required_fields=required,
        no_empty=not args.allow_empty,
        language=args.language,
    )

    print(f"\nFilter results:")
    print(f"  Kept: {stats['kept']}")
    for reason, count in sorted(stats["filtered"].items(), key=lambda x: -x[1]):
        print(f"  Removed ({reason}): {count}")

    if args.output:
        _save_jsonl(filtered, Path(args.output))
        print(f"\nSaved to: {args.output}")


def cmd_stats(args):
    """Show dataset statistics."""
    from dsprep.formats import load, load_as_conversations
    from dsprep.stats import dataset_stats, conversation_stats, format_stats_report

    if args.conversations:
        conversations = load_as_conversations(args.file, args.format)
        stats = conversation_stats(conversations)
    else:
        entries = load(args.file, args.format)
        stats = dataset_stats(entries)

    report = format_stats_report(stats)
    print(report)

    if args.json:
        # Remove sample from JSON output (not serializable summary)
        out = {k: v for k, v in stats.items() if k != "sample"}
        print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_sample(args):
    """Show random samples from the dataset."""
    import random
    from dsprep.formats import load

    entries = load(args.file, args.format)
    n = min(args.n, len(entries))
    samples = random.sample(entries, n)

    for i, entry in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"  Sample {i+1}/{n}")
        print(f"{'='*50}")
        print(json.dumps(entry, indent=2, ensure_ascii=False))


def cmd_head(args):
    """Show first N entries of a dataset."""
    from dsprep.formats import load

    entries = load(args.file, args.format)
    n = min(args.n, len(entries))

    for i in range(n):
        print(json.dumps(entries[i], ensure_ascii=False))

    print(f"\n({n}/{len(entries)} entries shown)")


def cmd_merge(args):
    """Merge multiple dataset files."""
    from dsprep.formats import load, _save_jsonl

    all_entries = []
    for path in args.files:
        entries = load(path, args.format)
        all_entries.extend(entries)
        print(f"  {Path(path).name}: {len(entries)} entries")

    print(f"\nTotal: {len(all_entries)} entries")

    if args.output:
        _save_jsonl(all_entries, Path(args.output))
        print(f"Saved to: {args.output}")


def cmd_split(args):
    """Split dataset into train/val/test sets."""
    import random
    from dsprep.formats import load, _save_jsonl

    entries = load(args.file, args.format)
    random.seed(args.seed)
    random.shuffle(entries)

    total = len(entries)
    test_size = int(total * args.test_ratio)
    val_size = int(total * args.val_ratio)
    train_size = total - test_size - val_size

    train = entries[:train_size]
    val = entries[train_size:train_size + val_size]
    test = entries[train_size + val_size:]

    output_dir = Path(args.output_dir)
    _save_jsonl(train, output_dir / "train.jsonl")
    _save_jsonl(val, output_dir / "val.jsonl")
    _save_jsonl(test, output_dir / "test.jsonl")

    print(f"Split {total} entries:")
    print(f"  Train: {len(train)} ({len(train)/total:.1%})")
    print(f"  Val:   {len(val)} ({len(val)/total:.1%})")
    print(f"  Test:  {len(test)} ({len(test)/total:.1%})")
    print(f"\nSaved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        prog="dsprep",
        description="Training dataset preparation toolkit"
    )
    sub = parser.add_subparsers(dest="command")

    # convert
    p = sub.add_parser("convert", help="Convert between formats")
    p.add_argument("input", help="Input file")
    p.add_argument("output", help="Output file")
    p.add_argument("--format", "-f", default="chatml", choices=["alpaca", "sharegpt", "chatml", "completion", "jsonl"],
                   help="Target format (default: chatml)")
    p.add_argument("--source-format", "-s", default=None, help="Source format (auto-detect if omitted)")

    # detect
    p = sub.add_parser("detect", help="Detect dataset format")
    p.add_argument("file", help="Dataset file")

    # dedup
    p = sub.add_parser("dedup", help="Remove duplicates")
    p.add_argument("file", help="Dataset file")
    p.add_argument("--output", "-o", default=None, help="Output file")
    p.add_argument("--key", "-k", default=None, help="Field to use as dedup key")
    p.add_argument("--keep", default="first", choices=["first", "last"])
    p.add_argument("--format", default=None, help="Source format")

    # filter
    p = sub.add_parser("filter", help="Filter by quality criteria")
    p.add_argument("file", help="Dataset file")
    p.add_argument("--output", "-o", default=None, help="Output file")
    p.add_argument("--min-length", type=int, default=None, help="Min total text length")
    p.add_argument("--max-length", type=int, default=None, help="Max total text length")
    p.add_argument("--required-fields", default=None, help="Comma-separated required fields")
    p.add_argument("--allow-empty", action="store_true", help="Allow empty string values")
    p.add_argument("--language", default=None, help="Filter by language (zh/en/ja)")
    p.add_argument("--format", default=None, help="Source format")

    # stats
    p = sub.add_parser("stats", help="Show dataset statistics")
    p.add_argument("file", help="Dataset file")
    p.add_argument("--conversations", "-c", action="store_true", help="Show conversation-level stats")
    p.add_argument("--json", "-j", action="store_true", help="Also output as JSON")
    p.add_argument("--format", default=None, help="Source format")

    # sample
    p = sub.add_parser("sample", help="Show random samples")
    p.add_argument("file", help="Dataset file")
    p.add_argument("-n", type=int, default=5, help="Number of samples")
    p.add_argument("--format", default=None, help="Source format")

    # head
    p = sub.add_parser("head", help="Show first N entries")
    p.add_argument("file", help="Dataset file")
    p.add_argument("-n", type=int, default=10, help="Number of entries")
    p.add_argument("--format", default=None, help="Source format")

    # merge
    p = sub.add_parser("merge", help="Merge multiple files")
    p.add_argument("files", nargs="+", help="Input files")
    p.add_argument("--output", "-o", required=True, help="Output file")
    p.add_argument("--format", default=None, help="Source format")

    # split
    p = sub.add_parser("split", help="Split into train/val/test")
    p.add_argument("file", help="Dataset file")
    p.add_argument("--output-dir", "-o", default="split", help="Output directory")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--format", default=None, help="Source format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "convert": cmd_convert,
        "detect": cmd_detect,
        "dedup": cmd_dedup,
        "filter": cmd_filter,
        "stats": cmd_stats,
        "sample": cmd_sample,
        "head": cmd_head,
        "merge": cmd_merge,
        "split": cmd_split,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
