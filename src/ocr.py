"""
Tesseract OCR integration for word-level text and bounding box extraction.

This module provides the spatial "word map" that the grounding engine uses
to locate VLM-extracted values on the page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pytesseract
from pytesseract import Output

from src.config import settings
from src.preprocessing import enhance_for_ocr
from src.schemas import BoundingBox

logger = logging.getLogger(__name__)

# Configure Tesseract path if set
if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class OCRWord:
    """A single word detected by Tesseract with its location and confidence."""

    text: str
    bbox: BoundingBox
    confidence: float  # 0.0 to 1.0
    block_num: int = 0
    line_num: int = 0
    word_num: int = 0


def get_word_map(image: np.ndarray, page: int = 0) -> list[OCRWord]:
    """
    Run Tesseract OCR on an image and extract all detected words with
    their bounding boxes and confidence scores.

    Args:
        image: BGR or grayscale image as numpy array.
        page: Page number to set on bounding boxes (for multi-page docs).

    Returns:
        List of OCRWord objects, filtered by confidence threshold.
    """
    # Enhance the image for better OCR results
    enhanced = enhance_for_ocr(image)

    # Run Tesseract with detailed output
    data = pytesseract.image_to_data(enhanced, output_type=Output.DICT)

    words: list[OCRWord] = []
    n_entries = len(data["text"])

    for i in range(n_entries):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        # Skip empty strings and low-confidence detections
        if not text or conf < settings.ocr_confidence_threshold:
            continue

        word = OCRWord(
            text=text,
            bbox=BoundingBox(
                x=float(data["left"][i]),
                y=float(data["top"][i]),
                width=float(data["width"][i]),
                height=float(data["height"][i]),
                page=page,
            ),
            confidence=conf / 100.0,
            block_num=data["block_num"][i],
            line_num=data["line_num"][i],
            word_num=data["word_num"][i],
        )
        words.append(word)

    logger.info(f"OCR detected {len(words)} words (page {page})")
    return words


def get_full_text(image: np.ndarray) -> str:
    """
    Get the full OCR text from an image (without bounding boxes).
    Useful for quick text checks or fallback processing.
    """
    enhanced = enhance_for_ocr(image)
    return pytesseract.image_to_string(enhanced).strip()


def merge_bounding_boxes(bboxes: list[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into a single box that encompasses all of them.

    Used to create a combined bounding box for multi-word values
    (e.g., a vendor name like "Acme Corp Ltd").
    """
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")

    if len(bboxes) == 1:
        return bboxes[0]

    x_min = min(b.x for b in bboxes)
    y_min = min(b.y for b in bboxes)
    x_max = max(b.x + b.width for b in bboxes)
    y_max = max(b.y + b.height for b in bboxes)

    return BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min,
        page=bboxes[0].page,
    )
