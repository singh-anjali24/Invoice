"""
Tests for the grounding engine (fuzzy matching VLM output → OCR bounding boxes).
"""

import pytest

from src.grounding import (
    ground_value,
    ground_numeric_value,
    GroundingResult,
)
from src.ocr import OCRWord
from src.schemas import BoundingBox


def _make_word(text: str, x: float = 0, y: float = 0, w: float = 50, h: float = 20) -> OCRWord:
    """Helper to create a test OCRWord."""
    return OCRWord(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=0.9,
    )


def _sample_word_map() -> list[OCRWord]:
    """Create a realistic word map simulating OCR output from an invoice."""
    return [
        _make_word("Invoice", x=50, y=50),
        _make_word("Number:", x=120, y=50),
        _make_word("INV-2026-00847", x=220, y=50),
        _make_word("Date:", x=50, y=80),
        _make_word("July", x=100, y=80),
        _make_word("10,", x=140, y=80),
        _make_word("2026", x=170, y=80),
        _make_word("ACME", x=50, y=120),
        _make_word("TECHNOLOGIES", x=110, y=120),
        _make_word("INC.", x=230, y=120),
        _make_word("Total:", x=350, y=400),
        _make_word("$5,845.50", x=420, y=400),
        _make_word("USD", x=500, y=400),
    ]


class TestGroundSingleWord:
    def test_exact_match(self):
        word_map = _sample_word_map()
        result = ground_value("INV-2026-00847", word_map)
        assert result.match_score > 0.9
        assert result.bbox is not None
        assert result.bbox.x == 220

    def test_case_insensitive(self):
        word_map = _sample_word_map()
        result = ground_value("inv-2026-00847", word_map)
        assert result.match_score > 0.8

    def test_no_match(self):
        word_map = _sample_word_map()
        result = ground_value("NONEXISTENT-VALUE", word_map)
        assert result.match_score == 0.0
        assert result.bbox is None


class TestGroundMultiWord:
    def test_multi_word_match(self):
        word_map = _sample_word_map()
        result = ground_value("ACME TECHNOLOGIES INC.", word_map)
        assert result.match_score > 0.7
        assert result.bbox is not None
        # Merged bounding box should span all 3 words
        assert result.bbox.width > 50  # Wider than a single word

    def test_partial_match(self):
        word_map = _sample_word_map()
        result = ground_value("ACME TECHNOLOGIES", word_map)
        assert result.match_score > 0.7


class TestGroundNumericValue:
    def test_exact_amount(self):
        word_map = _sample_word_map()
        # "$5,845.50" is in the word map
        result = ground_numeric_value(5845.50, word_map, threshold=0.6)
        assert result.match_score > 0.5

    def test_integer(self):
        word_map = [_make_word("2026", x=100, y=100)]
        result = ground_numeric_value(2026, word_map)
        assert result.match_score > 0.9

    def test_null_value(self):
        word_map = _sample_word_map()
        result = ground_numeric_value(None, word_map)
        assert result.match_score == 0.0


class TestGroundEmpty:
    def test_empty_value(self):
        result = ground_value("", _sample_word_map())
        assert result.match_score == 0.0

    def test_empty_word_map(self):
        result = ground_value("test", [])
        assert result.match_score == 0.0

    def test_null_value(self):
        result = ground_value(None, _sample_word_map())
        assert result.match_score == 0.0
