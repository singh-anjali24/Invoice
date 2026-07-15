"""
Validation layer for extracted invoice data.

Performs arithmetic and consistency checks that feed into the
confidence scoring. These checks catch real-world errors like
hallucinated totals, wrong date formats, and mismatched currency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from src.schemas import InvoiceExtractionResult

logger = logging.getLogger(__name__)

# Valid ISO 4217 currency codes (common subset)
VALID_CURRENCIES = {
    "USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY",
    "SEK", "NZD", "MXN", "SGD", "HKD", "NOK", "KRW", "TRY", "BRL",
    "ZAR", "DKK", "PLN", "THB", "AED", "MYR", "PHP", "CZK", "ILS",
    "CLP", "PKR", "BDT", "LKR", "NGN", "EGP", "SAR", "QAR", "KWD",
}


@dataclass
class ValidationResult:
    """Aggregated validation results for the entire extraction."""

    # Per-field validation scores (0.0 = failed, 1.0 = passed)
    field_scores: dict[str, float] = field(default_factory=dict)

    # Human-readable warnings
    warnings: list[str] = field(default_factory=list)

    # Overall validation pass rate
    overall_score: float = 0.0


def validate_extraction(invoice: InvoiceExtractionResult) -> ValidationResult:
    """
    Run all validation checks on the extracted invoice data.

    Returns a ValidationResult with per-field scores and warnings.
    """
    result = ValidationResult()

    # 1. Required fields present
    _check_required_fields(invoice, result)

    # 2. Date format validation
    _check_date_format(invoice, result)

    # 3. Arithmetic validation (line totals → grand total)
    _check_arithmetic(invoice, result)

    # 4. Currency validation
    _check_currency(invoice, result)

    # 5. Line item consistency
    _check_line_items(invoice, result)

    # Compute overall score
    if result.field_scores:
        result.overall_score = sum(result.field_scores.values()) / len(result.field_scores)
    else:
        result.overall_score = 0.0

    logger.info(
        f"Validation complete: score={result.overall_score:.2f}, "
        f"warnings={len(result.warnings)}"
    )

    return result


def _check_required_fields(invoice: InvoiceExtractionResult, result: ValidationResult):
    """Check that all required fields have non-null values."""
    required_fields = {
        "invoice_number": invoice.invoice_number.value,
        "invoice_date": invoice.invoice_date.value,
        "vendor_name": invoice.vendor_name.value,
        "total_amount": invoice.total_amount.value,
    }

    for field_name, value in required_fields.items():
        if value is None:
            result.field_scores[f"{field_name}_present"] = 0.0
            result.warnings.append(f"Required field '{field_name}' is null")
        else:
            result.field_scores[f"{field_name}_present"] = 1.0


def _check_date_format(invoice: InvoiceExtractionResult, result: ValidationResult):
    """Validate that dates are in ISO 8601 format (YYYY-MM-DD)."""
    date_fields = {
        "invoice_date": invoice.invoice_date.value,
        "due_date": invoice.due_date.value,
    }

    for field_name, value in date_fields.items():
        if value is None:
            # Optional field being null is fine
            if field_name != "invoice_date":
                result.field_scores[f"{field_name}_format"] = 1.0
            continue

        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            # Sanity check: year should be reasonable
            if 1990 <= parsed.year <= 2030:
                result.field_scores[f"{field_name}_format"] = 1.0
            else:
                result.field_scores[f"{field_name}_format"] = 0.5
                result.warnings.append(
                    f"Date '{value}' for '{field_name}' has unusual year: {parsed.year}"
                )
        except ValueError:
            result.field_scores[f"{field_name}_format"] = 0.0
            result.warnings.append(
                f"Date '{value}' for '{field_name}' is not valid ISO 8601 (YYYY-MM-DD)"
            )


def _check_arithmetic(invoice: InvoiceExtractionResult, result: ValidationResult):
    """
    Check that line item totals are consistent with the grand total.

    This is one of the strongest validation signals: if sum(line_totals)
    doesn't approximately equal total_amount, something is wrong.
    """
    if invoice.total_amount.value is None:
        result.field_scores["arithmetic"] = 0.0
        return

    if not invoice.line_items:
        # No line items to check against — can't validate
        result.field_scores["arithmetic"] = 0.5
        result.warnings.append("No line items to cross-check against total")
        return

    # Sum line totals
    line_sum = 0.0
    all_have_totals = True
    for item in invoice.line_items:
        if item.line_total.value is not None:
            line_sum += item.line_total.value
        else:
            all_have_totals = False

    if not all_have_totals:
        result.field_scores["arithmetic"] = 0.5
        result.warnings.append("Some line items are missing line_total values")
        return

    total = invoice.total_amount.value

    # Allow tolerance for tax, discounts, rounding
    # Total should be >= line_sum (because tax is added)
    # But total shouldn't be more than 2x line_sum (unreasonable tax)
    if total == 0:
        result.field_scores["arithmetic"] = 0.5
        return

    ratio = line_sum / total if total != 0 else 0

    if 0.70 <= ratio <= 1.05:
        # Line sum is close to total (within tax/discount range)
        result.field_scores["arithmetic"] = 1.0
    elif 0.50 <= ratio <= 1.30:
        # Somewhat off but plausible
        result.field_scores["arithmetic"] = 0.6
        result.warnings.append(
            f"Line items sum ({line_sum:.2f}) differs from total ({total:.2f}) "
            f"by {abs(line_sum - total):.2f}"
        )
    else:
        # Way off — likely an error
        result.field_scores["arithmetic"] = 0.2
        result.warnings.append(
            f"Arithmetic mismatch: line items sum to {line_sum:.2f} "
            f"but total is {total:.2f}"
        )


def _check_currency(invoice: InvoiceExtractionResult, result: ValidationResult):
    """Validate the currency code."""
    currency = invoice.currency.value
    if currency is None:
        result.field_scores["currency_valid"] = 0.0
        result.warnings.append("Currency is null")
        return

    if currency.upper() in VALID_CURRENCIES:
        result.field_scores["currency_valid"] = 1.0
    else:
        result.field_scores["currency_valid"] = 0.3
        result.warnings.append(f"Unknown currency code: '{currency}'")


def _check_line_items(invoice: InvoiceExtractionResult, result: ValidationResult):
    """Check individual line item consistency (qty × unit_price ≈ line_total)."""
    if not invoice.line_items:
        result.field_scores["line_item_consistency"] = 0.5
        return

    consistent_count = 0
    total_checked = 0

    for i, item in enumerate(invoice.line_items):
        qty = item.quantity.value
        price = item.unit_price.value
        total = item.line_total.value

        if qty is not None and price is not None and total is not None:
            total_checked += 1
            expected = qty * price
            # Allow small rounding tolerance
            if abs(expected - total) <= max(0.02 * abs(total), 0.01):
                consistent_count += 1
            else:
                result.warnings.append(
                    f"Line item {i + 1}: {qty} × {price:.2f} = {expected:.2f} "
                    f"≠ {total:.2f} (line_total)"
                )

    if total_checked > 0:
        result.field_scores["line_item_consistency"] = consistent_count / total_checked
    else:
        result.field_scores["line_item_consistency"] = 0.5
