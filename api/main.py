"""
FastAPI application for invoice data extraction.

Provides a REST API with:
- POST /extract — Upload an invoice and get structured JSON back
- GET  /health  — Health check endpoint
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.extractor import extract_invoice
from src.preprocessing import SUPPORTED_EXTENSIONS
from src.schemas import InvoiceExtractionResult

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Invoice Data Extraction API",
    description=(
        "Extract structured data from scanned invoices (PDF/image) using a "
        "hybrid VLM + OCR pipeline with confidence scoring and bounding boxes."
    ),
    version="1.0.0",
)

# Enable CORS for local development / Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed MIME types → file extensions
ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}


# ── Endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": settings.gemini_model,
        "version": "1.0.0",
    }


@app.post("/extract", response_model=InvoiceExtractionResult)
async def extract_invoice_endpoint(
    file: UploadFile = File(..., description="Invoice file (PDF, PNG, JPG, TIFF)")
):
    """
    Extract structured data from a scanned invoice.

    Upload a PDF or image file and receive a structured JSON response
    with all extracted fields, confidence scores, and bounding boxes.

    **Supported formats:** PDF, PNG, JPG/JPEG, TIFF, BMP, WebP

    **Max file size:** 10 MB
    """
    # ── Validate file type ──────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        # Try to infer from filename extension
        if file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {content_type} ({ext}). "
                    f"Supported: PDF, PNG, JPG, TIFF, BMP, WebP",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}",
            )

    # ── Read file content ───────────────────────────────────────────────
    content = await file.read()

    # Validate file size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content) / (1024 * 1024):.1f} MB. "
            f"Maximum: {settings.max_file_size_mb} MB",
        )

    # ── Save to temp file ───────────────────────────────────────────────
    # Use UUID filename to prevent directory traversal attacks
    ext = Path(file.filename or "upload.png").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        ext = ALLOWED_MIME_TYPES.get(content_type, ".png")

    temp_dir = Path(tempfile.gettempdir()) / "invoice_extraction"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}{ext}"

    try:
        temp_path.write_bytes(content)
        logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")

        # ── Run extraction ──────────────────────────────────────────────
        result = await extract_invoice(str(temp_path))

        # Update source_file to use the original filename
        if result.metadata:
            result.metadata.source_file = file.filename or "uploaded_file"

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}",
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


# ── Run with: uvicorn api.main:app --reload ─────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
