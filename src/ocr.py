"""
PaddleOCR integration for word-level text and bounding box extraction.

This module provides the spatial "word map" that the grounding engine uses
to locate VLM-extracted values on the page. PaddleOCR uses deep learning
for superior accuracy on invoices, tables, and multi-language documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.preprocessing import enhance_for_ocr
from src.schemas import BoundingBox

logger = logging.getLogger(__name__)

# Lazy-load the PaddleOCR model (loaded once, reused across calls)
_ocr_engine = None


def _get_ocr_engine():
    """Get or initialize the PaddleOCR engine (singleton)."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=True,  # Will automatically fallback to CPU if no GPU
        )
        logger.info("PaddleOCR engine initialized")
    return _ocr_engine


@dataclass
class OCRWord:
    """A single word detected by PaddleOCR with its location and confidence."""

    text: str
    bbox: BoundingBox
    confidence: float  # 0.0 to 1.0
    block_num: int = 0
    line_num: int = 0
    word_num: int = 0


def _quad_to_bbox(quad: list[list[float]], page: int = 0) -> BoundingBox:
    """
    Convert PaddleOCR's quadrilateral (4 corner points) to our
    rectangular BoundingBox format.

    PaddleOCR returns: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    We need: {x, y, width, height}
    """
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return BoundingBox(
        x=float(x_min),
        y=float(y_min),
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        page=page,
    )


def _split_line_into_words(
    text: str,
    quad: list[list[float]],
    confidence: float,
    line_num: int,
    page: int = 0,
) -> list[OCRWord]:
    """
    PaddleOCR returns line-level results (e.g., 'ACME TECHNOLOGIES INC.').
    Our grounding engine expects word-level results. This function splits
    a line into individual words, distributing the bounding box proportionally.
    """
    words_text = text.strip().split()
    if not words_text:
        return []

    # If it's a single word, no splitting needed
    if len(words_text) == 1:
        return [
            OCRWord(
                text=words_text[0],
                bbox=_quad_to_bbox(quad, page),
                confidence=confidence,
                line_num=line_num,
                word_num=0,
            )
        ]

    # For multi-word lines, distribute the bounding box proportionally
    line_bbox = _quad_to_bbox(quad, page)
    total_chars = sum(len(w) for w in words_text)

    result = []
    x_cursor = line_bbox.x

    for i, word in enumerate(words_text):
        # Proportional width based on character count
        word_width = (len(word) / total_chars) * line_bbox.width

        result.append(
            OCRWord(
                text=word,
                bbox=BoundingBox(
                    x=float(x_cursor),
                    y=line_bbox.y,
                    width=float(word_width),
                    height=line_bbox.height,
                    page=page,
                ),
                confidence=confidence,
                line_num=line_num,
                word_num=i,
            )
        )
        x_cursor += word_width

    return result


def get_word_map(image: np.ndarray, page: int = 0) -> list[OCRWord]:
    """
    Run PaddleOCR on an image and extract all detected words with
    their bounding boxes and confidence scores.

    Args:
        image: BGR or grayscale image as numpy array.
        page: Page number to set on bounding boxes (for multi-page docs).

    Returns:
        List of OCRWord objects, filtered by confidence threshold.
    """
    # Enhance the image for better OCR results
    enhanced = enhance_for_ocr(image)

    # Run PaddleOCR
    ocr = _get_ocr_engine()
    results = ocr.ocr(enhanced, cls=True)

    words: list[OCRWord] = []
    conf_threshold = settings.ocr_confidence_threshold / 100.0

    # results is a list of pages; each page is a list of line detections
    if results is None or len(results) == 0:
        logger.warning("PaddleOCR returned no results")
        return words

    page_results = results[0]  # First (and usually only) page
    if page_results is None:
        logger.warning("PaddleOCR returned None for page")
        return words

    for line_num, line in enumerate(page_results):
        quad = line[0]            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = line[1][0]         # Detected text string
        conf = float(line[1][1])  # Confidence score (0.0 to 1.0)

        # Skip low-confidence detections
        if conf < conf_threshold:
            continue

        # Split line-level result into individual words
        line_words = _split_line_into_words(
            text=text,
            quad=quad,
            confidence=conf,
            line_num=line_num,
            page=page,
        )
        words.extend(line_words)

    logger.info(f"OCR detected {len(words)} words (page {page})")
    return words


def get_full_text(image: np.ndarray) -> str:
    """
    Get the full OCR text from an image (without bounding boxes).
    Useful for quick text checks or fallback processing.
    """
    enhanced = enhance_for_ocr(image)
    ocr = _get_ocr_engine()
    results = ocr.ocr(enhanced, cls=True)

    if results is None or len(results) == 0 or results[0] is None:
        return ""

    lines = [line[1][0] for line in results[0]]
    return "\n".join(lines)


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
