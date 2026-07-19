# Technical Write-Up: Scanned Invoice Data Extraction System

## 1. Architecture Decision

### The Problem

Extracting structured data from scanned invoices presents two simultaneous challenges:

1. **Semantic Understanding** — Identifying what each piece of text *means* (is "2026-07-10" the invoice date or the due date? Is "ACME Corp" the vendor or the buyer?)
2. **Spatial Localization** — Knowing *where* on the page each value was found (bounding boxes)

No single technology excels at both. Traditional OCR (PaddleOCR) provides excellent spatial data but has zero semantic understanding. Vision Language Models (VLMs like Gemini/Qwen2) understand document semantics brilliantly but cannot return pixel-accurate coordinates.

### The Solution: Hybrid VLM + OCR Pipeline

I designed a hybrid architecture that plays to each technology's strengths:

```
┌─────────────┐     ┌───────────────┐
│  Gemini /    │     │  PaddleOCR    │
│  Qwen2 VLM   │     │  (Spatial)    │
│  (Semantic)  │     │               │
└──────┬──────┘     └───────┬───────┘
       │                    │
       └────────┬───────────┘
                │
       ┌────────▼────────┐
       │ Grounding Engine │  ← Fuzzy match to bridge the gap
       └────────┬────────┘
                │
       ┌────────▼────────┐
       │   Validation +   │
       │   Confidence     │
       └─────────────────┘
```

**The Vision Language Model (Gemini or Qwen2)** receives the invoice image directly and extracts all fields semantically. It understands layout, normalizes dates to ISO 8601, infers currency from symbols, and identifies vendor vs. buyer. This runs as a single API call (or local inference) with structured output (JSON schema constrained).

**PaddleOCR** runs in parallel on the same image, producing a "word map" — every detected text fragment with its pixel-level bounding box and per-character confidence score. PaddleOCR was chosen over Tesseract due to its superior accuracy with deep-learning-based text detection.

**The Grounding Engine** bridges the two: for each value the VLM extracted, it searches the OCR word map using fuzzy string matching (Levenshtein distance via `rapidfuzz`) to find where that value physically appears on the page. This produces the bounding boxes the assignment requires.

### Why Not Alternatives?

| Approach | Rejected Because |
|---|---|
| VLM only | Cannot produce pixel-accurate bounding boxes |
| OCR + regex rules | Zero semantic understanding; breaks on layout changes |
| Google Document AI | Pre-built service — doesn't demonstrate engineering capability |
| Fine-tuned LayoutLM | Requires training data and GPU; overkill for this scope |

---

## 2. Confidence Scoring Methodology

### Philosophy

A confidence score should answer: **"How likely is this extracted value to be correct?"**

A meaningful confidence score must be based on *observable evidence*, not arbitrary assignment. I use a composite of three independent signals, each measuring a different aspect of extraction quality.

### The Formula

```
confidence = w₁ × OCR_confidence + w₂ × grounding_match + w₃ × validation_score
```

Where:
- `w₁ = 0.30` (OCR signal weight)
- `w₂ = 0.35` (Grounding signal weight)
- `w₃ = 0.35` (Validation signal weight)

### Signal Definitions

#### Signal 1: OCR Confidence (weight: 0.30)

**Source:** PaddleOCR's per-word confidence score (0.0–1.0).

**What it measures:** How clearly the text was readable at the pixel level. A blurry, skewed, or poorly-scanned word will have low OCR confidence regardless of what the VLM thinks it says.

**Why 0.30 weight:** OCR confidence is a useful but noisy signal. PaddleOCR may give high confidence to incorrectly recognized characters or low confidence to perfectly readable text in unusual fonts. It's informative but shouldn't dominate.

#### Signal 2: Grounding Match Score (weight: 0.35)

**Source:** Fuzzy string matching (Levenshtein ratio) between the VLM's extracted value and the closest matching text in the OCR word map.

**What it measures:** Whether the VLM's output actually exists on the page. This is the primary **hallucination detector**. If Gemini returns `vendor_name = "Acme Technologies Inc."` but OCR found the words "ACME", "TECHNOLOGIES", and "INC." on the page with a combined fuzzy match score of 0.95, we have strong evidence the extraction is correct.

Conversely, if the VLM returns a value that can't be found anywhere in the OCR text, the grounding score drops to 0.0, and we apply an additional 0.6x penalty to the overall confidence — flagging a likely hallucination.

**Why 0.35 weight:** This is the most distinctive signal. High grounding means the VLM and OCR independently agree on the value. It's the strongest indicator of correctness.

#### Signal 3: Validation Score (weight: 0.35)

**Source:** Arithmetic and logical consistency checks.

**What it measures:** Whether the extracted data is internally consistent. Checks include:

1. **Arithmetic validation:** Does `sum(line_item.line_total)` approximately equal `total_amount`? (Allowing tolerance for tax/discounts)
2. **Line item consistency:** Does `quantity × unit_price ≈ line_total` for each item?
3. **Date format:** Is the date valid ISO 8601?
4. **Currency validity:** Is the currency a recognized ISO 4217 code?
5. **Required field presence:** Are required fields non-null?

**Why 0.35 weight:** Validation catches errors that neither OCR nor grounding can detect alone. If the VLM extracts a total of $5,845.50 and line items sum to $5,400.00, the difference ($445.50) is plausibly tax — validation passes. But if line items sum to $50.00 and the total is $5,845.50, something is clearly wrong — validation fails.

### Interpretation Guide

| Score Range | Label | Meaning |
|---|---|---|
| 0.80 – 1.00 | Very High | All three signals agree strongly. The value is almost certainly correct. |
| 0.60 – 0.80 | High | Strong agreement on most signals. Minor uncertainty in one area. |
| 0.30 – 0.60 | Medium | Mixed signals. The value is plausible but should be reviewed. |
| 0.00 – 0.30 | Low | Significant disagreement. The value may be hallucinated or incorrectly extracted. |

### Hallucination Penalty

If a field has a non-null value but the grounding engine fails to find it anywhere in the OCR word map (i.e., `bounding_box = null`), the composite confidence is multiplied by 0.6. This explicitly penalizes values that the VLM "sees" but OCR cannot confirm, which is a strong indicator of hallucination.

### Example

For a field `vendor_name = "ACME TECHNOLOGIES INC."`:

| Signal | Score | Reasoning |
|---|---|---|
| OCR Confidence | 0.91 | PaddleOCR read all three words with high confidence |
| Grounding Match | 0.95 | Fuzzy match found "ACME TECHNOLOGIES INC." in OCR with 95% similarity |
| Validation | 1.00 | Required field is present (check passed) |
| **Composite** | **0.30(0.91) + 0.35(0.95) + 0.35(1.00) = 0.956** | Very high confidence |

---

## 3. Bounding Box Strategy

### How Grounding Works

1. **Tokenization:** The VLM-extracted value is split into tokens (e.g., `"ACME TECHNOLOGIES INC."` → `["acme", "technologies", "inc."]`).

2. **Single-word matching:** For single-token values (e.g., invoice numbers), we use `rapidfuzz.process.extractOne()` to find the closest match in the entire OCR word map with O(n) complexity.

3. **Multi-word matching:** For multi-token values, we use a sliding window over the OCR word map. We compare the VLM value against every consecutive N-word sequence in the OCR output, using both exact ratio and token-sort ratio to handle word order variations.

4. **Numeric handling:** Numbers get special treatment. We generate multiple search variants (e.g., `5845.50` → `["5845.50", "5,845.50", "$5,845.50", "5845"]`) because OCR may format numbers differently than the VLM normalizes them.

5. **Box merging:** When a value spans multiple OCR words, the individual bounding boxes are merged into a single encompassing rectangle using min/max coordinates.

### Edge Cases

- **Value not found in OCR:** `bounding_box` is set to `null` and confidence is penalized. This honestly signals to the consumer that we couldn't locate the value on the page.
- **OCR splits differently than VLM:** The wider-window matching (±1-2 words) handles cases where OCR splits or merges words differently.
- **Currency symbols:** `$`, `₹`, `€` may be recognized as separate tokens by OCR. The numeric grounding variants handle this.

---

## 4. Error Handling & Robustness

### Graceful Degradation

The system is designed to never crash on bad input — it degrades gracefully:

- **Unreadable scan:** VLM may still extract values (it's more robust than OCR for poor images). Grounding will fail, resulting in null bounding boxes and lower confidence scores — but the extraction still returns results.
- **Missing fields:** Optional fields return `null` (never omitted), matching the assignment specification.
- **VLM hallucination:** Caught by grounding mismatch (can't find value in OCR) → confidence penalized.
- **Arithmetic errors:** Caught by validation layer → recorded as warnings in metadata.

### Parallel Execution

The VLM API call and OCR processing run concurrently via `asyncio.gather()`. Since they're independent operations, this cuts total processing time nearly in half compared to sequential execution.

---

## 5. Limitations & Future Improvements

### Current Limitations

1. **Single-page only:** The current implementation processes only the first page. Multi-page invoices would need page-level aggregation logic.
2. **Grounding accuracy:** Fuzzy matching works well for most cases but can fail on very short values (e.g., a single digit) or highly formatted text (e.g., barcodes).
3. **API dependency:** Requires Gemini API access. Offline operation would need a local VLM (e.g., LLaVA).
4. **Confidence calibration:** The weights (0.30, 0.35, 0.35) are based on reasoning about signal reliability, not empirical calibration on a labeled dataset. With access to ground-truth data, these could be optimized.

### Future Improvements

1. **Multi-page support:** Aggregate results across pages, handling split tables.
2. **Confidence calibration:** Collect ground-truth extractions and use logistic regression to learn optimal weights.
3. **Template caching:** For recurring vendors, cache successful extraction patterns to improve speed and accuracy.
4. **Human-in-the-loop:** Flag low-confidence extractions for manual review via a web UI.
5. **Batch processing:** Support async job queues (Celery) for high-volume invoice processing.

---

## 6. Technology Choices

| Component | Technology | Rationale |
|---|---|---|
| VLM | Gemini / Qwen2 | Swappable backends: Cloud API (Gemini) or Local Open-Source (Qwen2) for 100% privacy |
| OCR | PaddleOCR | Superior accuracy vs Tesseract, deep-learning based |
| Schema | Pydantic V2 | Type-safe JSON schema with validation, used by both VLM and API |
| Fuzzy Matching | rapidfuzz | C-optimized Levenshtein, 10x faster than fuzzywuzzy |
| API | FastAPI | Async-native, auto-generated docs, Pydantic integration |
| Image Processing | OpenCV + Pillow | Industry standard for preprocessing (deskew, threshold, denoise) |

---

## 7. Design Decisions

### Why Generic `ExtractedField[T]`?

Instead of flat JSON with parallel arrays for values, confidences, and bounding boxes, I wrapped each field in a generic `ExtractedField[T]` type:

```json
{
  "invoice_number": {
    "value": "INV-001",
    "confidence": 0.94,
    "bounding_box": {"x": 220, "y": 50, "width": 130, "height": 20}
  }
}
```

This is self-documenting — every field carries its own metadata. A consumer can check `field.confidence` and `field.bounding_box` without needing a lookup table.

### Why Structured Output (not free-form prompting)?

Gemini's `response_schema` parameter constrains the model to output valid JSON matching our Pydantic schema. This eliminates:
- JSON parsing errors
- Missing fields
- Wrong field names
- Type mismatches

The model can only return data that conforms to our schema, which drastically reduces post-processing complexity.
