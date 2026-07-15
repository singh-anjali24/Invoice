"""
Image preprocessing for invoice extraction.

Handles:
- PDF to image conversion (via pdf2image / Poppler)
- Image loading and normalization
- Enhancement for better OCR accuracy (grayscale, thresholding, deskew)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def load_document(file_path: str | Path) -> list[np.ndarray]:
    """
    Load a document (PDF or image) and return a list of page images as
    numpy arrays (BGR format, suitable for OpenCV).

    Args:
        file_path: Path to the invoice file (PDF or image).

    Returns:
        List of images, one per page. Single-page documents return a
        list with one element.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return _load_pdf(path)
    else:
        return _load_image(path)


def _load_pdf(path: Path) -> list[np.ndarray]:
    """Convert PDF pages to images using pdf2image."""
    try:
        from pdf2image import convert_from_path

        kwargs = {"dpi": 300}
        if settings.poppler_path:
            kwargs["poppler_path"] = settings.poppler_path

        pil_images = convert_from_path(str(path), **kwargs)
        images = []
        for pil_img in pil_images:
            # Convert PIL (RGB) to OpenCV (BGR)
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            images.append(bgr)

        logger.info(f"Loaded PDF with {len(images)} page(s): {path.name}")
        return images

    except ImportError:
        raise RuntimeError(
            "pdf2image is required for PDF processing. "
            "Install it with: pip install pdf2image\n"
            "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
        )


def _load_image(path: Path) -> list[np.ndarray]:
    """Load a single image file."""
    img = cv2.imread(str(path))
    if img is None:
        # Try with PIL as fallback (handles more formats)
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    logger.info(f"Loaded image: {path.name} ({img.shape[1]}x{img.shape[0]})")
    return [img]


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.

    Steps:
    1. Convert to grayscale
    2. Apply adaptive thresholding for better contrast
    3. Light denoising

    Args:
        image: BGR image as numpy array.

    Returns:
        Preprocessed grayscale image.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding (works well for varied lighting in scans)
    enhanced = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    return enhanced


def get_pil_image(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL Image (RGB)."""
    if len(image.shape) == 2:
        # Grayscale
        return Image.fromarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
