"""
Grounding engine: matches VLM-extracted values to OCR bounding boxes.

This is the bridge between semantic understanding (VLM) and spatial
localization (OCR). For each value the VLM extracted, we find where
it appears on the page by fuzzy-matching against the OCR word map.

Algorithm:
1. Tokenize the VLM-extracted value into searchable fragments
2. Use rapidfuzz (Levenshtein distance) to find matching words in the OCR map
3. Merge bounding boxes of all matched words
4. Return match quality score (used in confidence calculation)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from src.ocr import OCRWord, merge_bounding_boxes
from src.schemas import (
    BoundingBox,
    ExtractedField,
    InvoiceExtractionResult,
    LineItem,
    ExtractionMetadata,
    RawInvoiceExtraction,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding a single value against the OCR word map."""

    bbox: BoundingBox | None = None
    ocr_confidence: float = 0.0  # Average OCR confidence of matched words
    match_score: float = 0.0  # Fuzzy match quality (0.0–1.0)
    matched_words: list[OCRWord] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Split text into tokens for matching."""
    return _normalize_text(text).split()


def ground_value(
    value: str,
    word_map: list[OCRWord],
    threshold: float = 0.70,
) -> GroundingResult:
    """
    Find where a VLM-extracted value appears in the OCR word map.

    Strategy:
    - For short values (1-2 words): direct fuzzy match against individual words
    - For longer values: sliding window match against consecutive word groups

    Args:
        value: The value extracted by the VLM (e.g., "Acme Corp Ltd")
        word_map: All OCR-detected words with their bounding boxes
        threshold: Minimum fuzzy match score to accept (0.0–1.0)

    Returns:
        GroundingResult with bounding box and match quality
    """
    if not value or not word_map:
        return GroundingResult()

    value_normalized = _normalize_text(str(value))
    value_tokens = _tokenize(str(value))

    if not value_tokens:
        return GroundingResult()

    # Build a searchable index of OCR words
    ocr_texts = [w.text.lower().strip() for w in word_map]

    if len(value_tokens) == 1:
        # Single word: direct match
        return _match_single_word(value_normalized, word_map, ocr_texts, threshold)
    else:
        # Multi-word: sliding window
        return _match_multi_word(value_normalized, value_tokens, word_map, ocr_texts, threshold)


def _match_single_word(
    value: str,
    word_map: list[OCRWord],
    ocr_texts: list[str],
    threshold: float,
) -> GroundingResult:
    """Match a single-word value against the OCR word map."""
    # Use rapidfuzz to find the best match
    match = process.extractOne(
        value,
        ocr_texts,
        scorer=fuzz.ratio,
        score_cutoff=threshold * 100,
    )

    if match is None:
        return GroundingResult()

    matched_text, score, idx = match
    word = word_map[idx]

    return GroundingResult(
        bbox=word.bbox,
        ocr_confidence=word.confidence,
        match_score=score / 100.0,
        matched_words=[word],
    )


def _match_multi_word(
    value: str,
    value_tokens: list[str],
    word_map: list[OCRWord],
    ocr_texts: list[str],
    threshold: float,
) -> GroundingResult:
    """
    Match a multi-word value by finding the best consecutive word sequence
    in the OCR map that matches the value.
    """
    n_value_tokens = len(value_tokens)
    best_score = 0.0
    best_start = -1
    best_end = -1

    # Sliding window over OCR words
    for i in range(len(ocr_texts) - n_value_tokens + 1):
        window_text = " ".join(ocr_texts[i : i + n_value_tokens])
        score = fuzz.ratio(value, window_text) / 100.0

        if score > best_score:
            best_score = score
            best_start = i
            best_end = i + n_value_tokens

    # Also try wider windows (±1 word) to handle OCR splitting
    for extra in [1, 2]:
        for i in range(len(ocr_texts) - n_value_tokens - extra + 1):
            window_text = " ".join(ocr_texts[i : i + n_value_tokens + extra])
            score = fuzz.token_sort_ratio(value, window_text) / 100.0

            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + n_value_tokens + extra

    if best_score < threshold or best_start < 0:
        return GroundingResult()

    matched = word_map[best_start:best_end]
    bboxes = [w.bbox for w in matched]
    avg_confidence = sum(w.confidence for w in matched) / len(matched)

    return GroundingResult(
        bbox=merge_bounding_boxes(bboxes),
        ocr_confidence=avg_confidence,
        match_score=best_score,
        matched_words=matched,
    )


def ground_numeric_value(
    value: float | int,
    word_map: list[OCRWord],
    threshold: float = 0.80,
) -> GroundingResult:
    """
    Ground a numeric value (amounts, quantities) in the OCR word map.

    Numbers need special handling because OCR may format them differently
    (e.g., "1,500.00" vs "1500.00" or "$1500" vs "1500").
    """
    if value is None:
        return GroundingResult()

    value_str = str(value)

    # Generate multiple representations to search for
    search_variants = [value_str]

    if isinstance(value, float):
        # Try with 2 decimal places
        search_variants.append(f"{value:.2f}")
        # Try without decimals if it's a whole number
        if value == int(value):
            search_variants.append(str(int(value)))
        # Try with comma formatting
        if value >= 1000:
            search_variants.append(f"{value:,.2f}")
            search_variants.append(f"{value:,.0f}")

    if isinstance(value, int):
        search_variants.append(f"{value}.00")
        if value >= 1000:
            search_variants.append(f"{value:,}")

    # Remove duplicates
    search_variants = list(dict.fromkeys(search_variants))

    # Try each variant
    best_result = GroundingResult()
    for variant in search_variants:
        result = ground_value(variant, word_map, threshold)
        if result.match_score > best_result.match_score:
            best_result = result

    return best_result


# ── Date format variants ────────────────────────────────────────────────

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_ABBREVS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def ground_date_value(
    iso_date: str | None,
    word_map: list[OCRWord],
    threshold: float = 0.65,
) -> GroundingResult:
    """
    Ground a date value that was normalized to ISO 8601 by the VLM.

    The VLM outputs "2026-07-10" but the invoice might say
    "July 10, 2026" or "10/07/2026" or "10-Jul-2026". We generate
    all common date format variants and search for each.
    """
    if not iso_date or not word_map:
        return GroundingResult()

    # Parse the ISO date
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso_date)
    if not match:
        return ground_value(iso_date, word_map, threshold)

    year, month_str, day_str = match.groups()
    month = int(month_str)
    day = int(day_str)

    if not (1 <= month <= 12):
        return ground_value(iso_date, word_map, threshold)

    month_name = _MONTH_NAMES[month - 1]
    month_abbrev = _MONTH_ABBREVS[month - 1]
    day_no_pad = str(day)

    # Generate all common date format variants
    variants = [
        iso_date,                                    # 2026-07-10
        f"{month_name} {day_no_pad}, {year}",        # July 10, 2026
        f"{month_name} {day_str}, {year}",           # July 10, 2026
        f"{day_no_pad} {month_name} {year}",         # 10 July 2026
        f"{day_str} {month_name} {year}",            # 10 July 2026
        f"{month_abbrev} {day_no_pad}, {year}",      # Jul 10, 2026
        f"{month_abbrev} {day_str}, {year}",         # Jul 10, 2026
        f"{day_no_pad} {month_abbrev} {year}",       # 10 Jul 2026
        f"{month_str}/{day_str}/{year}",             # 07/10/2026
        f"{day_str}/{month_str}/{year}",             # 10/07/2026
        f"{month_str}-{day_str}-{year}",             # 07-10-2026
        f"{day_str}-{month_str}-{year}",             # 10-07-2026
        f"{day_str}.{month_str}.{year}",             # 10.07.2026
        f"{month_str}.{day_str}.{year}",             # 07.10.2026
    ]

    # Remove duplicates, preserve order
    variants = list(dict.fromkeys(variants))

    best_result = GroundingResult()
    for variant in variants:
        result = ground_value(variant, word_map, threshold)
        if result.match_score > best_result.match_score:
            best_result = result

    return best_result


# ── Currency symbol mapping ─────────────────────────────────────────────

_CURRENCY_SYMBOLS = {
    "USD": ["$", "US$", "USD"],
    "EUR": ["€", "EUR"],
    "GBP": ["£", "GBP"],
    "INR": ["₹", "INR", "Rs", "Rs."],
    "JPY": ["¥", "JPY"],
    "CAD": ["CA$", "CAD", "C$"],
    "AUD": ["A$", "AUD", "AU$"],
    "CHF": ["CHF", "Fr"],
    "CNY": ["¥", "CN¥", "CNY", "RMB"],
}


def ground_currency_value(
    currency_code: str | None,
    word_map: list[OCRWord],
    threshold: float = 0.65,
) -> GroundingResult:
    """
    Ground a currency code against the OCR word map.

    The VLM outputs "USD" but the invoice might only show "$".
    We search for both the code and its symbol variants.
    """
    if not currency_code or not word_map:
        return GroundingResult()

    code_upper = currency_code.upper()
    variants = [currency_code]

    # Add symbol variants for this currency
    if code_upper in _CURRENCY_SYMBOLS:
        variants.extend(_CURRENCY_SYMBOLS[code_upper])

    # Remove duplicates
    variants = list(dict.fromkeys(variants))

    best_result = GroundingResult()
    for variant in variants:
        result = ground_value(variant, word_map, threshold)
        if result.match_score > best_result.match_score:
            best_result = result

    return best_result


def ground_all_fields(
    raw: RawInvoiceExtraction,
    word_map: list[OCRWord],
    threshold: float = 0.70,
) -> InvoiceExtractionResult:
    """
    Ground all VLM-extracted fields against the OCR word map.

    Takes the raw VLM output and produces the final InvoiceExtractionResult
    with bounding boxes and initial confidence signals (OCR + grounding).
    Final confidence scoring happens in the confidence module.
    """
    logger.info("Grounding extracted fields against OCR word map...")

    def _ground_str(value: str | None) -> ExtractedField:
        if value is None:
            return ExtractedField(value=None, confidence=0.0, bounding_box=None)
        result = ground_value(str(value), word_map, threshold)
        return ExtractedField(
            value=value,
            confidence=0.0,  # Will be computed in confidence module
            bounding_box=result.bbox,
        )

    def _ground_num(value: float | int | None) -> ExtractedField:
        if value is None:
            return ExtractedField(value=None, confidence=0.0, bounding_box=None)
        result = ground_numeric_value(value, word_map, threshold)
        return ExtractedField(
            value=value,
            confidence=0.0,  # Will be computed in confidence module
            bounding_box=result.bbox,
        )

    def _ground_date(value: str | None) -> ExtractedField:
        if value is None:
            return ExtractedField(value=None, confidence=0.0, bounding_box=None)
        result = ground_date_value(value, word_map, threshold)
        return ExtractedField(
            value=value,
            confidence=0.0,
            bounding_box=result.bbox,
        )

    def _ground_currency(value: str | None) -> ExtractedField:
        if value is None:
            return ExtractedField(value=None, confidence=0.0, bounding_box=None)
        result = ground_currency_value(value, word_map, threshold)
        return ExtractedField(
            value=value,
            confidence=0.0,
            bounding_box=result.bbox,
        )

    # Ground each top-level field
    invoice = InvoiceExtractionResult(
        invoice_number=_ground_str(raw.invoice_number),
        invoice_date=_ground_date(raw.invoice_date),
        due_date=_ground_date(raw.due_date),
        vendor_name=_ground_str(raw.vendor_name),
        vendor_tax_id=_ground_str(raw.vendor_tax_id),
        buyer_name=_ground_str(raw.buyer_name),
        total_amount=_ground_num(raw.total_amount),
        currency=_ground_currency(raw.currency),
        line_items=[],
    )

    # Ground line items
    for raw_item in raw.line_items:
        grounded_item = LineItem(
            description=_ground_str(raw_item.description),
            quantity=_ground_num(raw_item.quantity),
            unit_price=_ground_num(raw_item.unit_price),
            line_total=_ground_num(raw_item.line_total),
        )
        invoice.line_items.append(grounded_item)

    grounded_count = sum(
        1
        for f in [
            invoice.invoice_number,
            invoice.invoice_date,
            invoice.due_date,
            invoice.vendor_name,
            invoice.vendor_tax_id,
            invoice.buyer_name,
            invoice.total_amount,
            invoice.currency,
        ]
        if f.bounding_box is not None
    )
    logger.info(f"Grounded {grounded_count}/8 top-level fields to bounding boxes")

    return invoice
