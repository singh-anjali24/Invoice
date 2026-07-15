"""
Command-line interface for invoice data extraction.

Usage:
    python cli.py invoice.pdf
    python cli.py invoice.png --output result.json
    python cli.py invoice.jpg --pretty
    python cli.py invoice.pdf --visualize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def visualize_result(image_path: str, result, output_path: str | None = None):
    """
    Overlay bounding boxes on the invoice image and save/display it.

    Draws colored rectangles around each detected field with labels.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image for visualization: {image_path}")
        return

    # Color map for different field types
    colors = {
        "invoice_number": (0, 255, 0),    # Green
        "invoice_date": (255, 165, 0),     # Orange
        "due_date": (255, 165, 0),         # Orange
        "vendor_name": (255, 0, 0),        # Blue (BGR)
        "vendor_tax_id": (255, 0, 0),      # Blue
        "buyer_name": (0, 165, 255),       # Orange-red
        "total_amount": (0, 0, 255),       # Red
        "currency": (128, 0, 128),         # Purple
        "line_item": (0, 200, 200),        # Yellow
    }

    def draw_field(name: str, field, color):
        if field.bounding_box is not None:
            bb = field.bounding_box
            x, y = int(bb.x), int(bb.y)
            w, h = int(bb.width), int(bb.height)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{name}: {field.value} ({field.confidence:.2f})"
            cv2.putText(
                img, label, (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    # Draw top-level fields
    for field_name in ["invoice_number", "invoice_date", "due_date", "vendor_name",
                        "vendor_tax_id", "buyer_name", "total_amount", "currency"]:
        field = getattr(result, field_name)
        draw_field(field_name, field, colors.get(field_name, (200, 200, 200)))

    # Draw line items
    for i, item in enumerate(result.line_items):
        color = colors["line_item"]
        for sub_field_name in ["description", "quantity", "unit_price", "line_total"]:
            sub_field = getattr(item, sub_field_name)
            draw_field(f"item_{i + 1}_{sub_field_name}", sub_field, color)

    # Save or display
    if output_path is None:
        output_path = str(Path(image_path).with_suffix(".annotated.png"))

    cv2.imwrite(output_path, img)
    print(f"\n🖼️  Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from scanned invoices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py invoice.pdf
  python cli.py invoice.png --output result.json
  python cli.py invoice.jpg --pretty --visualize
  python cli.py samples/invoice_1.png -v
        """,
    )
    parser.add_argument(
        "file",
        help="Path to the invoice file (PDF, PNG, JPG, TIFF)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save JSON output to a file instead of printing to stdout",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an annotated image with bounding boxes overlaid",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Output in flat downstream format (matches the assignment brief shape exactly)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)

    # Validate input file
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Run extraction
    from src.extractor import extract_invoice_sync

    print(f"📄 Processing: {input_path.name}")
    print("⏳ Running extraction pipeline...")

    try:
        result = extract_invoice_sync(str(input_path))
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}", file=sys.stderr)
        logging.getLogger(__name__).exception("Extraction error")
        sys.exit(1)

    # Format output — choose full or flat (downstream) format
    indent = 2 if args.pretty else None
    if args.flat:
        output_data = result.to_downstream_json()
        json_output = json.dumps(output_data, indent=indent, ensure_ascii=False)
    else:
        json_output = result.model_dump_json(indent=indent)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output, encoding="utf-8")
        print(f"\n✅ Results saved to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print("EXTRACTION RESULT" + (" (flat/downstream)" if args.flat else ""))
        print("=" * 60)
        print(json_output if not args.pretty else json.dumps(
            json.loads(json_output), indent=2, ensure_ascii=False
        ))

    # Visualization
    if args.visualize:
        visualize_result(str(input_path), result)

    # Summary
    print(f"\n📊 Summary:")
    print(f"   Invoice:  {result.invoice_number.value} (confidence: {result.invoice_number.confidence:.2f})")
    print(f"   Vendor:   {result.vendor_name.value} (confidence: {result.vendor_name.confidence:.2f})")
    print(f"   Total:    {result.currency.value} {result.total_amount.value} (confidence: {result.total_amount.confidence:.2f})")
    print(f"   Items:    {len(result.line_items)}")
    if result.metadata:
        print(f"   Time:     {result.metadata.processing_time_seconds:.2f}s")
        if result.metadata.validation_warnings:
            print(f"   Warnings: {len(result.metadata.validation_warnings)}")
            for w in result.metadata.validation_warnings:
                print(f"     ⚠️  {w}")


if __name__ == "__main__":
    main()
