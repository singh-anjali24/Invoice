# Invoice Data Extraction System

An AI-powered system that extracts structured data from scanned invoices (PDF/image) using a **hybrid VLM + OCR pipeline** with confidence scoring and bounding box localization.

## Architecture

```
Invoice (PDF/Image)
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  ← PDF→Image, enhance, deskew
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│  VLM   │ │Paddle  │  ← Run in PARALLEL
│(Gemini/│ │  OCR   │
│ Qwen2) │ └────────┘
└────────┘
    │          │
    ▼          ▼
┌─────────────────┐
│ Grounding Engine │  ← Fuzzy-match VLM values → OCR bounding boxes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation     │  ← Arithmetic, date format, currency checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Confidence      │  ← Composite scoring (OCR + grounding + validation)
│  Scoring         │
└────────┬────────┘
         │
         ▼
    Structured JSON
```

**Why hybrid?** The VLM (Gemini or Qwen2) understands invoice semantics (field identification, date normalization, currency inference) while PaddleOCR provides pixel-accurate bounding boxes. The grounding engine bridges the two via fuzzy string matching.

## Quick Start

### Option A: Docker (Recommended — zero system installs)

The only prerequisite is [Docker](https://www.docker.com/products/docker-desktop/). PaddleOCR, Poppler, and all dependencies are bundled inside the container.

```bash
# 1. Clone the repo
git clone <repo-url>
cd AI

# 2. Add your Gemini API key
copy .env.example .env
# Edit .env → set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)

# 3. Build & run
docker compose up --build

# API is now live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

To extract an invoice via the API:
```bash
curl -X POST http://localhost:8000/extract -F "file=@samples/invoice_us_acme.pdf"
```

To use the CLI inside the container:
```bash
docker compose run invoice-extractor python cli.py samples/invoice_us_acme.pdf --pretty
```

---

### Option B: Kaggle (Fully Open-Source, No API Keys)

For an environment with zero API keys and 100% data privacy, run the pipeline on Kaggle using a local open-source VLM (`Qwen2-VL-2B`) and PaddleOCR.

1. Go to Kaggle and create a **New Notebook**.
2. Go to **File → Import Notebook** and paste:
   ```
   https://github.com/singh-anjali24/Invoice/blob/main/notebooks/kaggle_demo.ipynb
   ```
3. Set **Accelerator** to **GPU T4 x2** and **Internet** to **On**.
4. Click **Run All**.

---

### Option C: Manual Install (without Docker)

#### Prerequisites

1. **Python 3.11+**
2. **PaddleOCR** — Installed via `pip install paddleocr paddlepaddle`
3. **Poppler** (for PDF support) — [Download for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
4. **Gemini API Key** (free) — [Get one here](https://aistudio.google.com/apikey) (if using the cloud VLM API backend)

#### Installation

```bash
# Clone the repository
git clone <repo-url>
cd AI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Generate Sample Invoices

```bash
python generate_samples.py
```

This creates 3 diverse invoices in `samples/`:
- `invoice_us_acme.pdf` — US corporate (USD, EIN)
- `invoice_india_tcs.pdf` — Indian GST (INR, GSTIN)
- `invoice_eu_mueller.pdf` — European (EUR, VAT)

### Usage

#### CLI

```bash
# Basic extraction
python cli.py samples/invoice_us_acme.pdf

# Pretty JSON output
python cli.py samples/invoice_us_acme.pdf --pretty

# Save to file
python cli.py samples/invoice_us_acme.pdf -o result.json --pretty

# Visualize bounding boxes on the invoice
python cli.py samples/invoice_us_acme.pdf --visualize

# Verbose logging
python cli.py samples/invoice_us_acme.pdf -v --pretty
```

#### REST API

```bash
# Start the server
uvicorn api.main:app --reload

# Open Swagger UI
# → http://localhost:8000/docs

# Upload via curl
curl -X POST http://localhost:8000/extract \
  -F "file=@samples/invoice_us_acme.pdf"
```

### Run Tests

```bash
pytest tests/ -v
```

## Output Schema

Every field is wrapped in `ExtractedField` with value, confidence, and bounding box:

```json
{
  "invoice_number": {
    "value": "INV-2026-00847",
    "confidence": 0.94,
    "bounding_box": {"x": 220, "y": 50, "width": 130, "height": 20, "page": 0}
  },
  "invoice_date": {
    "value": "2026-07-10",
    "confidence": 0.91,
    "bounding_box": {"x": 100, "y": 80, "width": 110, "height": 20, "page": 0}
  },
  "due_date": {
    "value": "2026-08-09",
    "confidence": 0.85,
    "bounding_box": null
  },
  "vendor_name": {
    "value": "ACME TECHNOLOGIES INC.",
    "confidence": 0.92,
    "bounding_box": {"x": 50, "y": 120, "width": 200, "height": 20, "page": 0}
  },
  "line_items": [
    {
      "description": {"value": "Cloud Hosting - Enterprise Plan", "confidence": 0.9, "bounding_box": {...}},
      "quantity": {"value": 1, "confidence": 0.95, "bounding_box": {...}},
      "unit_price": {"value": 2400.00, "confidence": 0.9, "bounding_box": {...}},
      "line_total": {"value": 2400.00, "confidence": 0.9, "bounding_box": {...}}
    }
  ],
  "total_amount": {"value": 5845.50, "confidence": 0.88, "bounding_box": {...}},
  "currency": {"value": "USD", "confidence": 0.95, "bounding_box": {...}},
  "metadata": {
    "source_file": "invoice_us_acme.pdf",
    "processing_time_seconds": 3.2,
    "validation_warnings": []
  }
}
```

## Confidence Scoring

See [WRITEUP.md](WRITEUP.md) for the full methodology. In brief:

```
confidence = 0.30 × OCR_confidence + 0.35 × grounding_match + 0.35 × validation_score
```

| Range | Meaning |
|---|---|
| 0.8–1.0 | Very High — strong multi-signal agreement |
| 0.6–0.8 | High — likely correct |
| 0.3–0.6 | Medium — plausible but uncertain |
| 0.0–0.3 | Low — likely incorrect or hallucinated |

## Project Structure

```
├── src/
│   ├── config.py          # Settings & environment management
│   ├── schemas.py         # Pydantic models (exact output shape)
│   ├── preprocessing.py   # PDF→Image, enhancement
│   ├── ocr.py             # PaddleOCR + bounding boxes
│   ├── vlm.py             # Gemini VLM integration
│   ├── grounding.py       # Fuzzy match VLM→OCR bounding boxes
│   ├── validation.py      # Arithmetic & consistency checks
│   ├── confidence.py      # Composite confidence scoring
│   └── extractor.py       # Main pipeline orchestrator
├── api/
│   └── main.py            # FastAPI REST API
├── cli.py                 # Command-line interface
├── tests/                 # Comprehensive test suite
├── samples/               # Generated sample invoices
├── WRITEUP.md             # Technical write-up
└── requirements.txt       # Dependencies
```

## License

MIT
