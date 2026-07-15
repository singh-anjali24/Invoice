"""
Gemini Vision Language Model integration for semantic invoice extraction.

This module sends the invoice image directly to Gemini 2.5 Flash, which
"reads" the invoice visually and returns structured field values. The VLM
handles layout understanding, date normalization, currency inference, and
other semantic tasks that traditional OCR cannot do.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from src.config import settings
from src.schemas import RawInvoiceExtraction, RawLineItem

logger = logging.getLogger(__name__)

# The prompt instructs the VLM exactly how to extract data
EXTRACTION_PROMPT = """You are an expert invoice data extraction system. Given this scanned invoice image, extract ALL fields into the provided JSON schema.

RULES — follow these precisely:
1. Dates MUST be normalized to ISO 8601 format: YYYY-MM-DD
   - "Jan 15, 2026" → "2026-01-15"
   - "15/01/2026" → "2026-01-15"
   - "2026-01-15" → "2026-01-15" (already correct)
2. If a field is NOT visible or CANNOT be determined from the invoice, set it to null. Do NOT guess.
3. For currency: infer from symbols (₹ = INR, $ = USD, € = EUR, £ = GBP) or from explicit text. If ambiguous, use the most likely currency based on context (vendor location, tax ID format, etc.)
4. For line_items: extract EVERY line item visible in the invoice table. Each line item must have description, quantity, unit_price, and line_total.
5. total_amount must be the FINAL payable total (after tax, discounts, etc.) — not the subtotal.
6. vendor_name is the company that ISSUED the invoice (the seller), not the buyer.
7. vendor_tax_id: look for GSTIN, VAT number, EIN, TIN, ABN, or any similar tax identifier. If multiple are present, use the most prominent one.
8. buyer_name is the company or person the invoice is addressed TO (the customer/client).
9. For quantity, if not explicitly stated (e.g., for service invoices), default to 1.
10. Ensure unit_price × quantity = line_total for each line item (within rounding tolerance)."""


async def extract_with_vlm(image: np.ndarray) -> RawInvoiceExtraction:
    """
    Send an invoice image to Gemini and get structured field extraction.

    Args:
        image: BGR image as numpy array.

    Returns:
        RawInvoiceExtraction with all fields populated by the VLM.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai is required. Install with: pip install google-genai"
        )

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and set it in your .env file."
        )

    # Initialize the Gemini client
    client = genai.Client(api_key=settings.gemini_api_key)

    # Convert numpy (BGR) to PNG bytes for the API
    pil_image = Image.fromarray(
        image[:, :, ::-1] if len(image.shape) == 3 else image
    )
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    logger.info(f"Sending image to {settings.gemini_model} for extraction...")

    # Call Gemini with structured output
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            EXTRACTION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RawInvoiceExtraction,
            temperature=0.1,  # Low temperature for deterministic extraction
        ),
    )

    # Parse the structured response
    result = RawInvoiceExtraction.model_validate_json(response.text)

    logger.info(
        f"VLM extracted: invoice_number={result.invoice_number}, "
        f"vendor={result.vendor_name}, "
        f"total={result.total_amount}, "
        f"line_items={len(result.line_items)}"
    )

    return result
