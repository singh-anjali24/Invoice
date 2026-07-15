"""
Generate realistic sample invoices for testing and demonstration.

Creates 3 diverse invoices:
1. US corporate style (USD)
2. Indian GST style (INR)
3. EU style (EUR, VAT)
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


SAMPLES_DIR = Path(__file__).parent / "samples"


def generate_invoice_1_us():
    """Generate a US-style corporate invoice (USD)."""
    output_path = SAMPLES_DIR / "invoice_us_acme.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Company Header
    header_style = ParagraphStyle(
        "CompanyHeader", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#1a5276"), spaceAfter=6,
    )
    elements.append(Paragraph("ACME TECHNOLOGIES INC.", header_style))

    sub_style = ParagraphStyle(
        "SubHeader", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#566573"), spaceAfter=2,
    )
    elements.append(Paragraph("742 Evergreen Terrace, Suite 400", sub_style))
    elements.append(Paragraph("San Francisco, CA 94102, USA", sub_style))
    elements.append(Paragraph("Phone: (415) 555-0198 | Email: billing@acmetech.com", sub_style))
    elements.append(Paragraph("EIN: 82-1234567", sub_style))
    elements.append(Spacer(1, 20))

    # Invoice Title
    inv_title = ParagraphStyle(
        "InvTitle", parent=styles["Title"],
        fontSize=28, textColor=colors.HexColor("#2e86c1"), spaceAfter=10,
    )
    elements.append(Paragraph("INVOICE", inv_title))

    # Invoice Details Table
    details_data = [
        ["Invoice Number:", "INV-2026-00847", "Bill To:", ""],
        ["Invoice Date:", "July 10, 2026", "", "GlobalRetail Solutions LLC"],
        ["Due Date:", "August 09, 2026", "", "1200 Commerce Blvd"],
        ["Payment Terms:", "Net 30", "", "Austin, TX 78701"],
    ]
    details_table = Table(details_data, colWidths=[110, 140, 60, 200])
    details_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a5276")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 0), (2, 0), colors.HexColor("#1a5276")),
        ("FONTNAME", (2, 0), (2, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 25))

    # Line Items Table
    items_data = [
        ["#", "Description", "Qty", "Unit Price", "Total"],
        ["1", "Cloud Hosting - Enterprise Plan (Monthly)", "1", "$2,400.00", "$2,400.00"],
        ["2", "API Gateway License", "3", "$350.00", "$1,050.00"],
        ["3", "Premium Support Package", "1", "$800.00", "$800.00"],
        ["4", "Data Analytics Module", "2", "$575.00", "$1,150.00"],
    ]
    items_table = Table(items_data, colWidths=[30, 250, 40, 90, 90])
    items_table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e86c1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        # Body
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5d8dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 15))

    # Totals
    totals_data = [
        ["", "", "Subtotal:", "$5,400.00"],
        ["", "", "Tax (8.25%):", "$445.50"],
        ["", "", "Total Due:", "$5,845.50"],
    ]
    totals_table = Table(totals_data, colWidths=[200, 100, 100, 100])
    totals_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (2, -1), (-1, -1), 12),
        ("TEXTCOLOR", (2, -1), (-1, -1), colors.HexColor("#1a5276")),
        ("LINEABOVE", (2, -1), (-1, -1), 1.5, colors.HexColor("#2e86c1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 30))

    # Footer
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#aab7b8"),
        alignment=1,
    )
    elements.append(Paragraph("Thank you for your business!", footer_style))
    elements.append(Paragraph(
        "Payment via wire transfer to: Chase Bank | Acct: 9876543210 | Routing: 021000021",
        footer_style,
    ))

    doc.build(elements)
    print(f"  ✅ Generated: {output_path}")
    return output_path


def generate_invoice_2_india():
    """Generate an Indian GST-style invoice (INR)."""
    output_path = SAMPLES_DIR / "invoice_india_tcs.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Company Header
    header_style = ParagraphStyle(
        "CompanyHeader", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#c0392b"), spaceAfter=4,
    )
    elements.append(Paragraph("BHARATH DIGITAL SERVICES PVT. LTD.", header_style))

    sub_style = ParagraphStyle(
        "SubHeader", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#566573"), spaceAfter=2,
    )
    elements.append(Paragraph("Rajiv Gandhi IT Park, Hinjewadi Phase 2", sub_style))
    elements.append(Paragraph("Pune, Maharashtra 411057, India", sub_style))
    elements.append(Paragraph("GSTIN: 27AABCB1234F1Z5", sub_style))
    elements.append(Paragraph("Phone: +91 20 6789 0123 | Email: accounts@bharathdigital.in", sub_style))
    elements.append(Spacer(1, 15))

    # Tax Invoice Title
    inv_title = ParagraphStyle(
        "InvTitle", parent=styles["Title"],
        fontSize=24, textColor=colors.HexColor("#c0392b"), spaceAfter=10,
    )
    elements.append(Paragraph("TAX INVOICE", inv_title))

    # Invoice Details
    details_data = [
        ["Invoice No:", "BDS/2026-27/0234", "Bill To:", ""],
        ["Date:", "05/07/2026", "", "Infosys Technologies Ltd"],
        ["Due Date:", "04/08/2026", "", "Electronics City, Hosur Road"],
        ["Place of Supply:", "Karnataka (29)", "", "Bengaluru, Karnataka 560100"],
        ["", "", "GSTIN:", "29AABCI5678L1Z8"],
    ]
    details_table = Table(details_data, colWidths=[100, 150, 55, 200])
    details_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#c0392b")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#c0392b")),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 20))

    # Line Items
    items_data = [
        ["Sr.", "Description", "HSN", "Qty", "Rate (₹)", "Amount (₹)"],
        ["1", "Software Development Services", "998314", "1", "3,50,000.00", "3,50,000.00"],
        ["2", "Cloud Infrastructure Setup", "998315", "1", "1,25,000.00", "1,25,000.00"],
        ["3", "Annual Maintenance Contract", "998316", "1", "75,000.00", "75,000.00"],
    ]
    items_table = Table(items_data, colWidths=[30, 200, 55, 35, 90, 95])
    items_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5d8dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdf2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 10))

    # Tax Breakdown + Totals
    totals_data = [
        ["", "Subtotal:", "₹5,50,000.00"],
        ["", "CGST (9%):", "₹49,500.00"],
        ["", "SGST (9%):", "₹49,500.00"],
        ["", "Total:", "₹6,49,000.00"],
    ]
    totals_table = Table(totals_data, colWidths=[280, 120, 110])
    totals_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (1, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (1, -1), (-1, -1), 12),
        ("TEXTCOLOR", (1, -1), (-1, -1), colors.HexColor("#c0392b")),
        ("LINEABOVE", (1, -1), (-1, -1), 1.5, colors.HexColor("#c0392b")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 20))

    # Amount in Words
    words_style = ParagraphStyle(
        "Words", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#1a5276"),
    )
    elements.append(Paragraph(
        "<b>Amount in Words:</b> Indian Rupees Six Lakh Forty-Nine Thousand Only",
        words_style,
    ))
    elements.append(Spacer(1, 20))

    # Footer
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#aab7b8"), alignment=1,
    )
    elements.append(Paragraph(
        "This is a computer-generated invoice and does not require a physical signature.",
        footer_style,
    ))

    doc.build(elements)
    print(f"  ✅ Generated: {output_path}")
    return output_path


def generate_invoice_3_eu():
    """Generate a EU-style invoice with VAT (EUR)."""
    output_path = SAMPLES_DIR / "invoice_eu_mueller.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Company Header
    header_style = ParagraphStyle(
        "CompanyHeader", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#1e8449"), spaceAfter=4,
    )
    elements.append(Paragraph("MÜLLER ENGINEERING GmbH", header_style))

    sub_style = ParagraphStyle(
        "SubHeader", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#566573"), spaceAfter=2,
    )
    elements.append(Paragraph("Friedrichstraße 123", sub_style))
    elements.append(Paragraph("10117 Berlin, Germany", sub_style))
    elements.append(Paragraph("VAT ID: DE 234567890", sub_style))
    elements.append(Paragraph("Tel: +49 30 1234 5678 | Email: rechnung@mueller-eng.de", sub_style))
    elements.append(Spacer(1, 15))

    # Invoice Title
    inv_title = ParagraphStyle(
        "InvTitle", parent=styles["Title"],
        fontSize=24, textColor=colors.HexColor("#1e8449"), spaceAfter=10,
    )
    elements.append(Paragraph("RECHNUNG / INVOICE", inv_title))

    # Invoice Details
    details_data = [
        ["Invoice No:", "ME-2026-1042", "Client:", ""],
        ["Date:", "2026-07-01", "", "TechnoVault B.V."],
        ["Due Date:", "2026-07-31", "", "Keizersgracht 456"],
        ["", "", "", "1016 GD Amsterdam, Netherlands"],
        ["", "", "VAT ID:", "NL 987654321B01"],
    ]
    details_table = Table(details_data, colWidths=[85, 145, 55, 220])
    details_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1e8449")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#1e8449")),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 20))

    # Line Items
    items_data = [
        ["Pos.", "Description", "Qty", "Unit Price (€)", "Total (€)"],
        ["1", "Mechanical Design - CAD Modeling (40 hrs)", "40", "95.00", "3,800.00"],
        ["2", "FEA Structural Analysis", "1", "2,200.00", "2,200.00"],
        ["3", "Prototype CNC Machining", "5", "480.00", "2,400.00"],
        ["4", "Technical Documentation Package", "1", "1,350.00", "1,350.00"],
        ["5", "Project Management", "1", "900.00", "900.00"],
    ]
    items_table = Table(items_data, colWidths=[35, 230, 35, 95, 95])
    items_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e8449")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5d8dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eafaf1")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 10))

    # Totals
    totals_data = [
        ["", "Netto / Subtotal:", "€10,650.00"],
        ["", "USt. / VAT (19%):", "€2,023.50"],
        ["", "Gesamt / Total:", "€12,673.50"],
    ]
    totals_table = Table(totals_data, colWidths=[280, 120, 100])
    totals_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (1, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (1, -1), (-1, -1), 12),
        ("TEXTCOLOR", (1, -1), (-1, -1), colors.HexColor("#1e8449")),
        ("LINEABOVE", (1, -1), (-1, -1), 1.5, colors.HexColor("#1e8449")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 25))

    # Bank Details
    bank_style = ParagraphStyle(
        "Bank", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#566573"),
    )
    elements.append(Paragraph("<b>Bank Details:</b>", bank_style))
    elements.append(Paragraph("Deutsche Bank AG | IBAN: DE89 3704 0044 0532 0130 00 | BIC: COBADEFFXXX", bank_style))
    elements.append(Spacer(1, 15))

    # Footer
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#aab7b8"), alignment=1,
    )
    elements.append(Paragraph(
        "Zahlbar innerhalb von 30 Tagen / Payment due within 30 days",
        footer_style,
    ))

    doc.build(elements)
    print(f"  ✅ Generated: {output_path}")
    return output_path


def generate_all_samples():
    """Generate all sample invoices."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print("📄 Generating sample invoices...")
    generate_invoice_1_us()
    generate_invoice_2_india()
    generate_invoice_3_eu()
    print(f"\n✅ All samples generated in: {SAMPLES_DIR}")


if __name__ == "__main__":
    generate_all_samples()
