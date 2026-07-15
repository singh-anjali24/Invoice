"""
Tests for the validation module.
"""

import pytest

from src.schemas import (
    ExtractedField,
    InvoiceExtractionResult,
    LineItem,
)
from src.validation import validate_extraction, ValidationResult


def _make_field(value, confidence=0.5):
    """Helper to create an ExtractedField."""
    return ExtractedField(value=value, confidence=confidence)


def _make_invoice(**overrides) -> InvoiceExtractionResult:
    """Helper to create a test invoice with sensible defaults."""
    defaults = dict(
        invoice_number=_make_field("INV-001"),
        invoice_date=_make_field("2026-07-10"),
        due_date=_make_field(None),
        vendor_name=_make_field("Test Corp"),
        vendor_tax_id=_make_field(None),
        buyer_name=_make_field(None),
        total_amount=_make_field(100.0),
        currency=_make_field("USD"),
        line_items=[],
    )
    defaults.update(overrides)
    return InvoiceExtractionResult(**defaults)


class TestRequiredFields:
    def test_all_required_present(self):
        invoice = _make_invoice()
        result = validate_extraction(invoice)
        assert result.field_scores["invoice_number_present"] == 1.0
        assert result.field_scores["vendor_name_present"] == 1.0

    def test_missing_required_field(self):
        invoice = _make_invoice(invoice_number=_make_field(None))
        result = validate_extraction(invoice)
        assert result.field_scores["invoice_number_present"] == 0.0
        assert any("invoice_number" in w for w in result.warnings)


class TestDateValidation:
    def test_valid_iso_date(self):
        invoice = _make_invoice(invoice_date=_make_field("2026-07-10"))
        result = validate_extraction(invoice)
        assert result.field_scores["invoice_date_format"] == 1.0

    def test_invalid_date_format(self):
        invoice = _make_invoice(invoice_date=_make_field("10/07/2026"))
        result = validate_extraction(invoice)
        assert result.field_scores["invoice_date_format"] == 0.0

    def test_null_optional_date(self):
        invoice = _make_invoice(due_date=_make_field(None))
        result = validate_extraction(invoice)
        assert result.field_scores.get("due_date_format", 1.0) == 1.0


class TestArithmeticValidation:
    def test_consistent_totals(self):
        items = [
            LineItem(
                description=_make_field("Item 1"),
                quantity=_make_field(2),
                unit_price=_make_field(25.0),
                line_total=_make_field(50.0),
            ),
            LineItem(
                description=_make_field("Item 2"),
                quantity=_make_field(1),
                unit_price=_make_field(50.0),
                line_total=_make_field(50.0),
            ),
        ]
        # Total = 100, sum of line totals = 100 → ratio = 1.0
        invoice = _make_invoice(total_amount=_make_field(100.0), line_items=items)
        result = validate_extraction(invoice)
        assert result.field_scores["arithmetic"] == 1.0

    def test_total_with_tax(self):
        """Total > sum of line items because of tax — should still pass."""
        items = [
            LineItem(line_total=_make_field(100.0)),
        ]
        # Total is 110 (with 10% tax), line sum is 100
        invoice = _make_invoice(total_amount=_make_field(110.0), line_items=items)
        result = validate_extraction(invoice)
        # ratio = 100/110 = 0.909 → within 0.70-1.05 → score 1.0
        assert result.field_scores["arithmetic"] == 1.0

    def test_arithmetic_mismatch(self):
        items = [
            LineItem(line_total=_make_field(50.0)),
        ]
        # Total is 500 but line items sum to 50 → way off
        invoice = _make_invoice(total_amount=_make_field(500.0), line_items=items)
        result = validate_extraction(invoice)
        assert result.field_scores["arithmetic"] < 0.5


class TestCurrencyValidation:
    def test_valid_currency(self):
        invoice = _make_invoice(currency=_make_field("USD"))
        result = validate_extraction(invoice)
        assert result.field_scores["currency_valid"] == 1.0

    def test_valid_inr(self):
        invoice = _make_invoice(currency=_make_field("INR"))
        result = validate_extraction(invoice)
        assert result.field_scores["currency_valid"] == 1.0

    def test_invalid_currency(self):
        invoice = _make_invoice(currency=_make_field("XYZ"))
        result = validate_extraction(invoice)
        assert result.field_scores["currency_valid"] == 0.3


class TestLineItemConsistency:
    def test_consistent_line_items(self):
        items = [
            LineItem(
                description=_make_field("Widget"),
                quantity=_make_field(3),
                unit_price=_make_field(10.0),
                line_total=_make_field(30.0),
            ),
        ]
        invoice = _make_invoice(total_amount=_make_field(30.0), line_items=items)
        result = validate_extraction(invoice)
        assert result.field_scores["line_item_consistency"] == 1.0

    def test_inconsistent_line_item(self):
        items = [
            LineItem(
                description=_make_field("Widget"),
                quantity=_make_field(3),
                unit_price=_make_field(10.0),
                line_total=_make_field(50.0),  # 3 × 10 ≠ 50
            ),
        ]
        invoice = _make_invoice(total_amount=_make_field(50.0), line_items=items)
        result = validate_extraction(invoice)
        assert result.field_scores["line_item_consistency"] == 0.0


class TestOverallScore:
    def test_overall_score_computed(self):
        invoice = _make_invoice()
        result = validate_extraction(invoice)
        assert 0.0 <= result.overall_score <= 1.0
