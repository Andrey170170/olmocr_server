## olmOCR HTTP API

This service exposes a simple HTTP API for OCRing PDFs and images using the olmOCR pipeline. It can run against a local vLLM instance (auto‑started on boot) or an external vLLM/OpenAI‑compatible server.

### Base URL

- Default: `http://localhost:8000`
- Configurable via container env `API_PORT`.

### Authentication

- This API has no built‑in auth. If configured to use an external inference server that requires an API key, that key is used server‑side and is not exposed to clients.

### Endpoints

#### GET /health

Readiness probe. Returns 200 only when the inference backend is ready.

Responses:
- 200 `{ "status": "ready" }`
- 503 `{ "status": "initializing" }`

Example:
```bash
curl -s http://localhost:8000/health
```

#### POST /process

OCR a single file (PDF, PNG, or JPEG).

Request:
- Multipart form-data with part name `file`.

Response 200 (Dolma-style document):
```json
{
  "id": "<sha1-of-text>",
  "text": "... full plain-text/markdown of document ...",
  "source": "olmocr",
  "added": "YYYY-MM-DD",
  "created": "YYYY-MM-DD",
  "metadata": {
    "Source-File": "<original filename>",
    "olmocr-version": "<semver>",
    "pdf-total-pages": 3,
    "total-input-tokens": 1234,
    "total-output-tokens": 567,
    "total-fallback-pages": 0
  },
  "attributes": {
    "pdf_page_numbers": [[0, 123, 1], [124, 456, 2]],
    "primary_language": ["en", "en"],
    "is_rotation_valid": [true, true],
    "rotation_correction": [0, 0],
    "is_table": [false, false],
    "is_diagram": [false, false]
  }
}
```

Errors:
- 400 `{ "detail": "Failed to process document" }`
- 503 `{ "detail": "Initializing" }` (startup not complete)

Examples:
```bash
curl -F file=@test_data/sample.pdf http://localhost:8000/process
```

Python (requests):
```python
import requests
with open("sample.pdf", "rb") as f:
    r = requests.post("http://localhost:8000/process", files={"file": ("sample.pdf", f, "application/pdf")})
    r.raise_for_status()
    doc = r.json()
```

JavaScript (fetch):
```javascript
const form = new FormData();
form.append("file", new Blob([fileBytes], { type: "application/pdf" }), "sample.pdf");
const res = await fetch("http://localhost:8000/process", { method: "POST", body: form });
if (!res.ok) throw new Error(await res.text());
const doc = await res.json();
```

#### POST /process_batch

OCR multiple files in one request.

Request:
- Multipart form-data with one or more `files` parts.

Response 200:
- JSON array mixing successful Dolma docs and per-file errors:
```json
[
  { "id": "...", "text": "...", "metadata": {"pdf-total-pages": 2}, "attributes": {"pdf_page_numbers": [[0,10,1],[11,20,2]]} },
  { "filename": "bad.pdf", "error": "failed" }
]
```

Example:
```bash
curl -F files=@doc1.pdf -F files=@doc2.pdf http://localhost:8000/process_batch
```

### Persistence (server-side)

For observability, the server writes outputs to a workspace on disk:
- JSONL: `${OLMOCR_WORKSPACE}/results/output_<uuid>.jsonl` (one line per processed input)
- Optional Markdown (when `OLMOCR_MARKDOWN=true`): `${OLMOCR_WORKSPACE}/markdown/<basename>.md`

Defaults:
- `OLMOCR_WORKSPACE=/workspace`
- `OLMOCR_MARKDOWN=false`

Note: This does not affect HTTP responses and is provided for offline inspection.

### Runtime configuration (env)

Core:
- `API_PORT` (default `8000`): API listen port
- `OLMOCR_WORKSPACE` (default `/workspace`)
- `OLMOCR_MARKDOWN` (default `false`)

Inference selection:
- `OLMOCR_SERVER` (unset => launch local vLLM)
- `OLMOCR_API_KEY` (optional; used when `OLMOCR_SERVER` requires Bearer auth)
- `OLMOCR_MODEL` (default `allenai/olmOCR-7B-0825-FP8`)

Local vLLM tuning (when `OLMOCR_SERVER` unset):
- `OLMOCR_PORT` (default `30024`) internal vLLM port
- `OLMOCR_GPU_MEMORY_UTILIZATION` (e.g. `0.80`)
- `OLMOCR_MAX_MODEL_LEN` (default `16384`)
- `OLMOCR_TENSOR_PARALLEL_SIZE` (default `1`)
- `OLMOCR_DATA_PARALLEL_SIZE` (default `1`)

Page processing knobs:
- `OLMOCR_TARGET_IMAGE_DIM` (default `1288`)
- `OLMOCR_MAX_PAGE_RETRIES` (default `8`)
- `OLMOCR_MAX_PAGE_ERROR_RATE` (default `0.004`)
- `OLMOCR_GUIDED_DECODING` (default `false`)

### File types & notes

- Accepts `.pdf`, `.png`, `.jpg`/`.jpeg`. Images are internally converted to PDF before processing.
- Very large files can increase latency linearly with page count.

### Docker quickstart

Build and run locally (GPU required for local vLLM):
```bash
docker build -t olmocr-api:latest .
docker run --gpus all -p 8000:8000 -e API_PORT=8000 -v $(pwd)/workspace:/workspace olmocr-api:latest
```

Health:
```bash
curl http://localhost:8000/health
```

### Client implementation tips

- Use multipart form-data and stream file uploads when possible.
- Treat non-200 responses as errors; 503 means retry after a short delay (service is initializing).
- The response JSON can be large; prefer streaming parsers for very long documents.
- For batch, handle mixed results: entries with `{ filename, error }` indicate per-file failures.


