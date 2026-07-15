"""
Composite confidence scoring for extracted invoice fields.

Confidence Score Methodology
============================

Each extracted field receives a confidence score from 0.0 to 1.0, computed
as a weighted combination of three independent signals:

    confidence = w1 × ocr_confidence + w2 × grounding_score + w3 × validation_score

Signal Definitions:
-------------------
1. OCR Confidence (weight: 0.30)
   Source: Tesseract's per-word confidence score.
   What it measures: How clearly the text was readable at the pixel level.
   Range: 0.0 (unreadable) to 1.0 (crystal clear).

2. Grounding Match Score (weight: 0.35)
   Source: Fuzzy string matching between VLM output and OCR text.
   What it measures: Whether the VLM's extracted value actually exists on the
   page. If the VLM "sees" something that OCR can't find, it may be hallucinated.
   Range: 0.0 (no match found) to 1.0 (exact match).

3. Validation Score (weight: 0.35)
   Source: Arithmetic and consistency checks.
   What it measures: Cross-field logical consistency (e.g., do line totals
   sum correctly? Is the date valid ISO 8601?).
   Range: 0.0 (all checks failed) to 1.0 (all checks passed).

Interpretation:
- 0.0–0.3: LOW — value likely incorrect or hallucinated
- 0.3–0.6: MEDIUM — value plausible but uncertain
- 0.6–0.8: HIGH — value likely correct
- 0.8–1.0: VERY HIGH — strong multi-signal agreement
"""

from __future__ import annotations

import logging

from src.config import settings
from src.grounding import ground_value, ground_numeric_value, ground_date_value, ground_currency_value, GroundingResult
from src.ocr import OCRWord
from src.schemas import ExtractedField, InvoiceExtractionResult, LineItem
from src.validation import ValidationResult

logger = logging.getLogger(__name__)


def _compute_field_confidence(
    field: ExtractedField,
    word_map: list[OCRWord],
    validation_score: float,
    is_numeric: bool = False,
    grounding_type: str = "string",
) -> float:
    """
    Compute the composite confidence for a single extracted field.

    Args:
        field: The extracted field (with value and optional bounding box)
        word_map: The OCR word map (for re-computing grounding if needed)
        validation_score: The validation score for this field (0.0–1.0)
        is_numeric: Whether the field value is numeric
        grounding_type: One of "string", "numeric", "date", "currency"

    Returns:
        Composite confidence score (0.0–1.0)
    """
    if field.value is None:
        return 0.0

    # Signal 1 & 2: OCR Confidence + Grounding Match Score
    # Use the appropriate grounding function based on field type
    if grounding_type == "date":
        grounding = ground_date_value(str(field.value), word_map)
    elif grounding_type == "currency":
        grounding = ground_currency_value(str(field.value), word_map)
    elif is_numeric or grounding_type == "numeric":
        grounding = ground_numeric_value(field.value, word_map)
    else:
        grounding = ground_value(str(field.value), word_map)

    ocr_conf = grounding.ocr_confidence if grounding.matched_words else 0.0

    # Signal 2: Grounding Match Score
    grounding_score = grounding.match_score

    # Signal 3: Validation Score (passed in)
    val_score = validation_score

    # Weighted combination
    confidence = (
        settings.weight_ocr * ocr_conf
        + settings.weight_grounding * grounding_score
        + settings.weight_validation * val_score
    )

    # Penalty: if no bounding box was found, reduce confidence
    # This penalizes potential hallucinations
    if field.bounding_box is None and field.value is not None:
        confidence *= 0.6

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, round(confidence, 3)))


def compute_confidences(
    invoice: InvoiceExtractionResult,
    validation: ValidationResult,
    word_map: list[OCRWord],
) -> InvoiceExtractionResult:
    """
    Compute confidence scores for all fields in the extraction result.

    Mutates the invoice in-place and returns it.
    """
    logger.info("Computing composite confidence scores...")

    # Map field names to their validation scores
    def _get_val_score(*keys: str) -> float:
        """Get the average validation score for the given keys."""
        scores = [validation.field_scores.get(k, 0.5) for k in keys]
        return sum(scores) / len(scores) if scores else 0.5

    # Top-level string fields
    invoice.invoice_number.confidence = _compute_field_confidence(
        invoice.invoice_number, word_map,
        _get_val_score("invoice_number_present"),
    )

    # Date fields — use date-aware grounding
    invoice.invoice_date.confidence = _compute_field_confidence(
        invoice.invoice_date, word_map,
        _get_val_score("invoice_date_present", "invoice_date_format"),
        grounding_type="date",
    )

    invoice.due_date.confidence = _compute_field_confidence(
        invoice.due_date, word_map,
        _get_val_score("due_date_format"),
        grounding_type="date",
    )

    invoice.vendor_name.confidence = _compute_field_confidence(
        invoice.vendor_name, word_map,
        _get_val_score("vendor_name_present"),
    )

    invoice.vendor_tax_id.confidence = _compute_field_confidence(
        invoice.vendor_tax_id, word_map,
        validation.overall_score,
    )

    invoice.buyer_name.confidence = _compute_field_confidence(
        invoice.buyer_name, word_map,
        validation.overall_score,
    )

    # Numeric fields
    invoice.total_amount.confidence = _compute_field_confidence(
        invoice.total_amount, word_map,
        _get_val_score("total_amount_present", "arithmetic"),
        is_numeric=True,
    )

    # Currency — use currency-aware grounding
    invoice.currency.confidence = _compute_field_confidence(
        invoice.currency, word_map,
        _get_val_score("currency_valid"),
        grounding_type="currency",
    )

    # Line items
    line_item_val = _get_val_score("line_item_consistency")
    for item in invoice.line_items:
        item.description.confidence = _compute_field_confidence(
            item.description, word_map, line_item_val,
        )
        item.quantity.confidence = _compute_field_confidence(
            item.quantity, word_map, line_item_val, is_numeric=True,
        )
        item.unit_price.confidence = _compute_field_confidence(
            item.unit_price, word_map, line_item_val, is_numeric=True,
        )
        item.line_total.confidence = _compute_field_confidence(
            item.line_total, word_map, line_item_val, is_numeric=True,
        )

    # Store validation warnings in metadata
    if invoice.metadata:
        invoice.metadata.validation_warnings = validation.warnings

    # Log summary
    avg_conf = _average_confidence(invoice)
    logger.info(f"Average confidence across all fields: {avg_conf:.3f}")

    return invoice


def _average_confidence(invoice: InvoiceExtractionResult) -> float:
    """Compute the average confidence across all top-level fields."""
    scores = [
        invoice.invoice_number.confidence,
        invoice.invoice_date.confidence,
        invoice.vendor_name.confidence,
        invoice.total_amount.confidence,
        invoice.currency.confidence,
    ]
    # Add optional fields only if they have values
    if invoice.due_date.value is not None:
        scores.append(invoice.due_date.confidence)
    if invoice.vendor_tax_id.value is not None:
        scores.append(invoice.vendor_tax_id.confidence)
    if invoice.buyer_name.value is not None:
        scores.append(invoice.buyer_name.confidence)

    return sum(scores) / len(scores) if scores else 0.0
