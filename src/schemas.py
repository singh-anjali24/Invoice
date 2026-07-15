"""
Pydantic models defining the exact JSON schema for invoice extraction output.

Design: Each extracted field is wrapped in ExtractedField[T] which bundles
the value, confidence score, and bounding box together. This makes the
output self-documenting and matches the assignment requirements precisely.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class BoundingBox(BaseModel):
    """
    Pixel-level bounding box for a detected value on the invoice image.

    Coordinates are relative to the top-left corner of the page image.
    """

    x: float = Field(..., description="Left edge in pixels")
    y: float = Field(..., description="Top edge in pixels")
    width: float = Field(..., ge=0, description="Width in pixels")
    height: float = Field(..., ge=0, description="Height in pixels")
    page: int = Field(default=0, ge=0, description="Page index (0-based)")


class ExtractedField(BaseModel, Generic[T]):
    """
    Wrapper for any extracted field, bundling the value with its
    confidence score and source location on the document.

    Confidence scoring methodology (explained in WRITEUP.md):
      - 0.0–0.3: Low confidence — value likely incorrect or hallucinated
      - 0.3–0.6: Medium — value plausible but uncertain
      - 0.6–0.8: High — value likely correct
      - 0.8–1.0: Very high — strong multi-signal agreement
    """

    value: T | None = Field(default=None, description="The extracted value, or null if not found")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)"
    )
    bounding_box: BoundingBox | None = Field(
        default=None, description="Approximate location on the page, or null if not locatable"
    )


class LineItem(BaseModel):
    """A single line item from the invoice."""

    description: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str]()
    )
    quantity: ExtractedField[int] = Field(
        default_factory=lambda: ExtractedField[int]()
    )
    unit_price: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField[float]()
    )
    line_total: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField[float]()
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process itself."""

    source_file: str = Field(..., description="Original filename")
    model_used: str = Field(default="gemini-2.5-flash", description="VLM model used")
    processing_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Total processing time"
    )
    image_width: int = Field(default=0, ge=0, description="Source image width in pixels")
    image_height: int = Field(default=0, ge=0, description="Source image height in pixels")
    ocr_words_detected: int = Field(
        default=0, ge=0, description="Number of words found by OCR"
    )
    validation_warnings: list[str] = Field(
        default_factory=list, description="Any validation issues found"
    )


class InvoiceExtractionResult(BaseModel):
    """
    Complete extraction result matching the assignment's required schema.

    Every field from the assignment brief is present. Optional fields
    use null (not omitted) when the value is not found, as specified.
    """

    invoice_number: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str](),
        description="Invoice number (required)",
    )
    invoice_date: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str](),
        description="Invoice date in ISO 8601 format YYYY-MM-DD (required)",
    )
    due_date: ExtractedField[str | None] = Field(
        default_factory=lambda: ExtractedField[str | None](),
        description="Due date in ISO 8601 format (optional)",
    )
    vendor_name: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str](),
        description="Issuing company name (required)",
    )
    vendor_tax_id: ExtractedField[str | None] = Field(
        default_factory=lambda: ExtractedField[str | None](),
        description="GSTIN / VAT / EIN tax identifier (optional)",
    )
    buyer_name: ExtractedField[str | None] = Field(
        default_factory=lambda: ExtractedField[str | None](),
        description="Buyer/customer name (optional)",
    )
    line_items: list[LineItem] = Field(
        default_factory=list, description="Individual line items"
    )
    total_amount: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField[float](),
        description="Final payable total amount (required)",
    )
    currency: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str](),
        description="Currency code e.g. USD, INR, EUR (required)",
    )
    metadata: ExtractionMetadata | None = Field(
        default=None, description="Processing metadata"
    )

    def to_downstream_json(self) -> dict:
        """
        Produce the exact JSON shape a downstream system would consume,
        matching the assignment brief precisely.

        This is the "flat" output format where each field has its value
        alongside a confidence score and bounding box. Missing optional
        fields are null, not omitted.

        Returns:
            dict matching the exact schema from the assignment brief.
        """

        def _bbox_dict(bbox: BoundingBox | None) -> dict | None:
            if bbox is None:
                return None
            return {
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height,
                "page": bbox.page,
            }

        def _field_dict(field: ExtractedField) -> dict:
            return {
                "value": field.value,
                "confidence": field.confidence,
                "bounding_box": _bbox_dict(field.bounding_box),
            }

        return {
            "invoice_number": _field_dict(self.invoice_number),
            "invoice_date": _field_dict(self.invoice_date),
            "due_date": _field_dict(self.due_date),
            "vendor_name": _field_dict(self.vendor_name),
            "vendor_tax_id": _field_dict(self.vendor_tax_id),
            "buyer_name": _field_dict(self.buyer_name),
            "line_items": [
                {
                    "description": item.description.value,
                    "quantity": item.quantity.value,
                    "unit_price": item.unit_price.value,
                    "line_total": item.line_total.value,
                    "confidence": {
                        "description": item.description.confidence,
                        "quantity": item.quantity.confidence,
                        "unit_price": item.unit_price.confidence,
                        "line_total": item.line_total.confidence,
                    },
                    "bounding_boxes": {
                        "description": _bbox_dict(item.description.bounding_box),
                        "quantity": _bbox_dict(item.quantity.bounding_box),
                        "unit_price": _bbox_dict(item.unit_price.bounding_box),
                        "line_total": _bbox_dict(item.line_total.bounding_box),
                    },
                }
                for item in self.line_items
            ],
            "total_amount": _field_dict(self.total_amount),
            "currency": _field_dict(self.currency),
        }


# ---------------------------------------------------------------------------
# Intermediate models used internally (VLM raw output before grounding)
# ---------------------------------------------------------------------------


class RawLineItem(BaseModel):
    """Line item as returned by the VLM (no bounding boxes yet)."""

    description: str | None = None
    quantity: int | None = None
    unit_price: float | None = None
    line_total: float | None = None


class RawInvoiceExtraction(BaseModel):
    """
    Flat extraction result from the VLM, before grounding and scoring.
    This schema is passed to Gemini's structured output mode.
    """

    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    vendor_name: str | None = None
    vendor_tax_id: str | None = None
    buyer_name: str | None = None
    line_items: list[RawLineItem] = Field(default_factory=list)
    total_amount: float | None = None
    currency: str | None = None
