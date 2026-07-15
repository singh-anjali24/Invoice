"""
Tests for Pydantic schemas — validates that the output JSON shape matches
the assignment requirements exactly.
"""

import json

import pytest
from pydantic import ValidationError

from src.schemas import (
    BoundingBox,
    ExtractedField,
    ExtractionMetadata,
    InvoiceExtractionResult,
    LineItem,
    RawInvoiceExtraction,
    RawLineItem,
)


class TestBoundingBox:
    """Tests for the BoundingBox model."""

    def test_valid_bounding_box(self):
        bb = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0, page=0)
        assert bb.x == 10.0
        assert bb.y == 20.0
        assert bb.width == 100.0
        assert bb.height == 50.0
        assert bb.page == 0

    def test_default_page(self):
        bb = BoundingBox(x=0, y=0, width=10, height=10)
        assert bb.page == 0

    def test_negative_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(x=0, y=0, width=-10, height=10)


class TestExtractedField:
    """Tests for the ExtractedField generic wrapper."""

    def test_string_field(self):
        field = ExtractedField[str](value="INV-001", confidence=0.95)
        assert field.value == "INV-001"
        assert field.confidence == 0.95
        assert field.bounding_box is None

    def test_float_field(self):
        field = ExtractedField[float](value=1234.56, confidence=0.8)
        assert field.value == 1234.56

    def test_null_value(self):
        field = ExtractedField[str](value=None, confidence=0.0)
        assert field.value is None
        # Null value should serialize to null, not be omitted
        data = field.model_dump()
        assert "value" in data
        assert data["value"] is None

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            ExtractedField[str](value="test", confidence=1.5)
        with pytest.raises(ValidationError):
            ExtractedField[str](value="test", confidence=-0.1)

    def test_with_bounding_box(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        field = ExtractedField[str](value="test", confidence=0.9, bounding_box=bb)
        assert field.bounding_box is not None
        assert field.bounding_box.x == 10


class TestLineItem:
    """Tests for the LineItem model."""

    def test_complete_line_item(self):
        item = LineItem(
            description=ExtractedField[str](value="Widget", confidence=0.9),
            quantity=ExtractedField[int](value=5, confidence=0.85),
            unit_price=ExtractedField[float](value=10.00, confidence=0.9),
            line_total=ExtractedField[float](value=50.00, confidence=0.88),
        )
        assert item.description.value == "Widget"
        assert item.quantity.value == 5

    def test_default_line_item(self):
        item = LineItem()
        assert item.description.value is None
        assert item.quantity.value is None


class TestInvoiceExtractionResult:
    """Tests for the complete extraction result schema."""

    def test_all_fields_present_in_json(self):
        """Assignment requirement: missing fields must be null, not omitted."""
        result = InvoiceExtractionResult()
        data = result.model_dump()

        # Every field must exist in the output
        required_keys = [
            "invoice_number", "invoice_date", "due_date",
            "vendor_name", "vendor_tax_id", "buyer_name",
            "line_items", "total_amount", "currency",
        ]
        for key in required_keys:
            assert key in data, f"Field '{key}' missing from output"

    def test_optional_fields_are_null_not_omitted(self):
        """Due date, vendor_tax_id, buyer_name should be null when absent."""
        result = InvoiceExtractionResult()
        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["due_date"]["value"] is None
        assert data["vendor_tax_id"]["value"] is None
        assert data["buyer_name"]["value"] is None

    def test_complete_result(self):
        """Test a fully populated result."""
        result = InvoiceExtractionResult(
            invoice_number=ExtractedField[str](value="INV-001", confidence=0.95),
            invoice_date=ExtractedField[str](value="2026-07-10", confidence=0.9),
            due_date=ExtractedField[str | None](value="2026-08-10", confidence=0.85),
            vendor_name=ExtractedField[str](value="Acme Corp", confidence=0.92),
            vendor_tax_id=ExtractedField[str | None](value="82-1234567", confidence=0.8),
            buyer_name=ExtractedField[str | None](value="GlobalRetail LLC", confidence=0.88),
            line_items=[
                LineItem(
                    description=ExtractedField[str](value="Cloud Hosting", confidence=0.9),
                    quantity=ExtractedField[int](value=1, confidence=0.95),
                    unit_price=ExtractedField[float](value=2400.00, confidence=0.9),
                    line_total=ExtractedField[float](value=2400.00, confidence=0.9),
                ),
            ],
            total_amount=ExtractedField[float](value=5845.50, confidence=0.88),
            currency=ExtractedField[str](value="USD", confidence=0.95),
            metadata=ExtractionMetadata(
                source_file="test.pdf",
                processing_time_seconds=2.5,
            ),
        )

        data = json.loads(result.model_dump_json())
        assert data["invoice_number"]["value"] == "INV-001"
        assert data["total_amount"]["value"] == 5845.50
        assert len(data["line_items"]) == 1
        assert data["line_items"][0]["description"]["value"] == "Cloud Hosting"

    def test_json_roundtrip(self):
        """Ensure result survives JSON serialization/deserialization."""
        result = InvoiceExtractionResult(
            invoice_number=ExtractedField[str](value="INV-999", confidence=0.5),
            vendor_name=ExtractedField[str](value="Test", confidence=0.5),
            total_amount=ExtractedField[float](value=100.0, confidence=0.5),
            currency=ExtractedField[str](value="USD", confidence=0.5),
        )
        json_str = result.model_dump_json()
        restored = InvoiceExtractionResult.model_validate_json(json_str)
        assert restored.invoice_number.value == "INV-999"


class TestRawInvoiceExtraction:
    """Tests for the intermediate VLM output schema."""

    def test_from_json(self):
        """Simulate Gemini's structured output."""
        raw_json = {
            "invoice_number": "INV-2026-00847",
            "invoice_date": "2026-07-10",
            "due_date": "2026-08-09",
            "vendor_name": "ACME TECHNOLOGIES INC.",
            "vendor_tax_id": "82-1234567",
            "buyer_name": "GlobalRetail Solutions LLC",
            "line_items": [
                {
                    "description": "Cloud Hosting - Enterprise Plan",
                    "quantity": 1,
                    "unit_price": 2400.00,
                    "line_total": 2400.00,
                },
                {
                    "description": "API Gateway License",
                    "quantity": 3,
                    "unit_price": 350.00,
                    "line_total": 1050.00,
                },
            ],
            "total_amount": 5845.50,
            "currency": "USD",
        }
        raw = RawInvoiceExtraction.model_validate(raw_json)
        assert raw.invoice_number == "INV-2026-00847"
        assert len(raw.line_items) == 2
        assert raw.total_amount == 5845.50

    def test_missing_optional_fields(self):
        """Optional fields should default to None."""
        raw = RawInvoiceExtraction(
            invoice_number="INV-001",
            vendor_name="Test",
            total_amount=100.0,
        )
        assert raw.due_date is None
        assert raw.vendor_tax_id is None
        assert raw.buyer_name is None
