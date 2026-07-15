"""
Main extraction pipeline — orchestrates all components.

Pipeline flow:
1. Preprocess (load & enhance image)
2. Run VLM extraction + OCR in parallel
3. Ground VLM outputs to OCR bounding boxes
4. Validate extracted data
5. Compute composite confidence scores
6. Return final structured result
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from src.config import settings
from src.confidence import compute_confidences
from src.grounding import ground_all_fields
from src.ocr import get_word_map
from src.preprocessing import load_document
from src.schemas import ExtractionMetadata, InvoiceExtractionResult
from src.validation import validate_extraction
from src.vlm import extract_with_vlm

logger = logging.getLogger(__name__)


async def extract_invoice(file_path: str | Path) -> InvoiceExtractionResult:
    """
    Extract structured data from a scanned invoice.

    This is the main entry point for the extraction pipeline. It orchestrates
    all components (VLM, OCR, grounding, validation, confidence scoring) and
    returns a complete InvoiceExtractionResult.

    Args:
        file_path: Path to the invoice file (PDF, PNG, JPG, etc.)

    Returns:
        InvoiceExtractionResult with all fields, confidence scores, and
        bounding boxes populated.

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is unsupported
        RuntimeError: If VLM or OCR processing fails
    """
    start_time = time.time()
    path = Path(file_path)

    logger.info(f"Starting extraction for: {path.name}")

    # ── Step 1: Preprocess ──────────────────────────────────────────────
    logger.info("Step 1/5: Loading and preprocessing document...")
    images = load_document(file_path)
    # For now, process only the first page (most invoices are single-page)
    image = images[0]
    height, width = image.shape[:2]
    logger.info(f"  Image size: {width}x{height} pixels")

    # ── Step 2: Run VLM + OCR in parallel ───────────────────────────────
    logger.info("Step 2/5: Running VLM extraction and OCR in parallel...")
    vlm_result, word_map = await asyncio.gather(
        extract_with_vlm(image),
        asyncio.to_thread(get_word_map, image, 0),
    )
    logger.info(
        f"  VLM returned {len(vlm_result.line_items)} line items, "
        f"OCR detected {len(word_map)} words"
    )

    # ── Step 3: Ground VLM outputs to OCR bounding boxes ────────────────
    logger.info("Step 3/5: Grounding VLM outputs to OCR bounding boxes...")
    invoice = ground_all_fields(vlm_result, word_map, settings.grounding_match_threshold)

    # ── Step 4: Validate extracted data ─────────────────────────────────
    logger.info("Step 4/5: Validating extracted data...")
    validation = validate_extraction(invoice)

    # ── Step 5: Compute confidence scores ───────────────────────────────
    logger.info("Step 5/5: Computing confidence scores...")
    invoice = compute_confidences(invoice, validation, word_map)

    # ── Attach metadata ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    invoice.metadata = ExtractionMetadata(
        source_file=path.name,
        model_used=settings.gemini_model,
        processing_time_seconds=round(elapsed, 2),
        image_width=width,
        image_height=height,
        ocr_words_detected=len(word_map),
        validation_warnings=validation.warnings,
    )

    logger.info(
        f"Extraction complete in {elapsed:.2f}s: "
        f"invoice #{invoice.invoice_number.value} "
        f"from {invoice.vendor_name.value} "
        f"for {invoice.currency.value} {invoice.total_amount.value}"
    )

    return invoice


def extract_invoice_sync(file_path: str | Path) -> InvoiceExtractionResult:
    """
    Synchronous wrapper for extract_invoice.
    Useful for CLI and simple scripts.
    """
    return asyncio.run(extract_invoice(file_path))
