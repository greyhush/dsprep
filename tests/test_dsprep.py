"""Tests for dsprep."""

import json
import os
import tempfile
import pytest

from dsprep.formats import detect_format, convert, load, load_as_conversations
from dsprep.dedup import deduplicate, find_duplicates
from dsprep.filters import filter_dataset, detect_language
from dsprep.stats import dataset_stats, conversation_stats


def _write_jsonl(entries, suffix=".jsonl"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


class TestFormatDetection:
    def test_alpaca(self):
        path = _write_jsonl([{"instruction": "hi", "input": "", "output": "hello"}])
        assert detect_format(path) == "alpaca"
        os.unlink(path)

    def test_sharegpt(self):
        path = _write_jsonl([{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}])
        assert detect_format(path) == "sharegpt"
        os.unlink(path)

    def test_chatml(self):
        path = _write_jsonl([{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}])
        assert detect_format(path) == "chatml"
        os.unlink(path)

    def test_completion(self):
        path = _write_jsonl([{"text": "hello world"}])
        assert detect_format(path) == "completion"
        os.unlink(path)


class TestConversion:
    def test_alpaca_to_chatml(self):
        alpaca = [{"instruction": "What is 1+1?", "input": "", "output": "2"}]
        path = _write_jsonl(alpaca)
        out_path = path + ".out.jsonl"

        count = convert(path, out_path, "alpaca", "chatml")
        assert count == 1

        result = load(out_path, "chatml")
        assert result[0]["messages"][0]["role"] == "user"
        assert "1+1" in result[0]["messages"][0]["content"]
        assert result[0]["messages"][1]["role"] == "assistant"
        assert result[0]["messages"][1]["content"] == "2"

        os.unlink(path)
        os.unlink(out_path)

    def test_sharegpt_to_alpaca(self):
        sgpt = [{"conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]}]
        path = _write_jsonl(sgpt)
        out_path = path + ".out.jsonl"

        count = convert(path, out_path, "sharegpt", "alpaca")
        assert count == 1

        result = load(out_path, "alpaca")
        assert result[0]["instruction"] == "Hello"
        assert result[0]["output"] == "Hi there"

        os.unlink(path)
        os.unlink(out_path)

    def test_roundtrip(self):
        """chatml -> sharegpt -> chatml should preserve content."""
        chatml = [{"messages": [
            {"role": "user", "content": "What is physics?"},
            {"role": "assistant", "content": "Physics is the study of matter and energy."},
        ]}]
        path = _write_jsonl(chatml)
        mid_path = path + ".sgpt.jsonl"
        out_path = path + ".out.jsonl"

        convert(path, mid_path, "chatml", "sharegpt")
        convert(mid_path, out_path, "sharegpt", "chatml")

        result = load(out_path, "chatml")
        assert result[0]["messages"][0]["content"] == "What is physics?"
        assert result[0]["messages"][1]["content"] == "Physics is the study of matter and energy."

        os.unlink(path)
        os.unlink(mid_path)
        os.unlink(out_path)


class TestDedup:
    def test_basic(self):
        entries = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "hello"},
        ]
        result, dupes = deduplicate(entries)
        assert len(result) == 2
        assert dupes == 1

    def test_with_key(self):
        entries = [
            {"id": 1, "text": "hello"},
            {"id": 1, "text": "different"},
            {"id": 2, "text": "world"},
        ]
        result, dupes = deduplicate(entries, key="id")
        assert len(result) == 2

    def test_keep_last(self):
        entries = [
            {"id": 1, "text": "first"},
            {"id": 1, "text": "last"},
        ]
        result, _ = deduplicate(entries, key="id", keep="last")
        assert result[0]["text"] == "last"

    def test_find_duplicates(self):
        entries = [
            {"text": "a"},
            {"text": "b"},
            {"text": "a"},
            {"text": "b"},
        ]
        groups = find_duplicates(entries)
        assert len(groups) == 2


class TestFilters:
    def test_min_length(self):
        entries = [{"text": "hi"}, {"text": "a long enough text for testing"}]
        result, stats = filter_dataset(entries, min_length=10)
        assert len(result) == 1
        assert stats["filtered"]["too_short"] == 1

    def test_required_fields(self):
        entries = [
            {"instruction": "do something", "output": "done"},
            {"instruction": "", "output": "done"},
        ]
        result, stats = filter_dataset(entries, required_fields=["instruction"])
        assert len(result) == 1

    def test_no_empty(self):
        entries = [
            {"text": "hello"},
            {"text": ""},
        ]
        result, stats = filter_dataset(entries)
        assert len(result) == 1

    def test_allow_empty(self):
        entries = [
            {"text": "hello"},
            {"text": ""},
        ]
        result, stats = filter_dataset(entries, no_empty=False)
        assert len(result) == 2

    def test_language_filter(self):
        entries = [
            {"text": "Hello world this is English"},
            {"text": "你好世界这是一个中文测试文本"},
        ]
        result, stats = filter_dataset(entries, language="en")
        assert len(result) == 1
        assert "Hello" in result[0]["text"]


class TestLanguageDetection:
    def test_english(self):
        assert detect_language("Hello world this is a test") == "en"

    def test_chinese(self):
        assert detect_language("你好世界这是一个测试") == "zh"

    def test_japanese(self):
        assert detect_language("こんにちは世界テストです") == "ja"


class TestStats:
    def test_basic(self):
        entries = [
            {"instruction": "hello", "output": "hi"},
            {"instruction": "world", "output": "earth"},
        ]
        stats = dataset_stats(entries)
        assert stats["count"] == 2
        assert "instruction" in stats["fields"]
        assert stats["text_lengths"]["min"] > 0

    def test_conversation_stats(self):
        convs = [
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hey"}, {"role": "assistant", "content": "hi"}],
        ]
        stats = conversation_stats(convs)
        assert stats["count"] == 2
        assert stats["turns"]["min"] == 2
        assert stats["system_present"] == 1
