import asyncio
import os
import tempfile
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

import olmocr.pipeline as pipeline
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png


def load_args_from_env() -> SimpleNamespace:
    """Build an args-like object from environment variables for reuse in pipeline calls."""

    def getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
        val = os.environ.get(name)
        return val if val is not None and val != "" else default

    def getenv_float(name: str, default: Optional[float]) -> Optional[float]:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else default

    def getenv_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else default

    return SimpleNamespace(
        # Inference server selection
        server=getenv_str("OLMOCR_SERVER"),
        api_key=getenv_str("OLMOCR_API_KEY"),
        model=getenv_str("OLMOCR_MODEL", "allenai/olmOCR-7B-0825-FP8"),
        # Local vLLM server knobs
        gpu_memory_utilization=getenv_float("OLMOCR_GPU_MEMORY_UTILIZATION", None),
        max_model_len=getenv_int("OLMOCR_MAX_MODEL_LEN", 16384),
        tensor_parallel_size=getenv_int("OLMOCR_TENSOR_PARALLEL_SIZE", 1),
        data_parallel_size=getenv_int("OLMOCR_DATA_PARALLEL_SIZE", 1),
        # Page processing knobs
        target_longest_image_dim=getenv_int("OLMOCR_TARGET_IMAGE_DIM", 1288),
        max_page_retries=getenv_int("OLMOCR_MAX_PAGE_RETRIES", 8),
        max_page_error_rate=float(os.environ.get("OLMOCR_MAX_PAGE_ERROR_RATE", "0.004")),
        guided_decoding=os.environ.get("OLMOCR_GUIDED_DECODING", "false").lower() == "true",
        # Workspace
        workspace=getenv_str("OLMOCR_WORKSPACE", "/workspace"),
        markdown=os.environ.get("OLMOCR_MARKDOWN", "false").lower() == "true",
        # Server port used by local vLLM
        port=int(os.environ.get("OLMOCR_PORT", "30024")),
    )


def set_pipeline_base_port_from_args(args: SimpleNamespace) -> None:
    """Set the pipeline's BASE_SERVER_PORT global based on args.port."""
    pipeline.BASE_SERVER_PORT = args.port


async def start_local_vllm_and_wait(args: SimpleNamespace) -> Tuple[asyncio.Task, asyncio.Semaphore]:
    """Download model if needed, start local vLLM server, wait for readiness, return (task, semaphore)."""
    set_pipeline_base_port_from_args(args)
    model_path_or_name = await pipeline.download_model(args.model)
    semaphore = asyncio.Semaphore(1)
    task = asyncio.create_task(pipeline.vllm_server_host(model_path_or_name, args, semaphore))
    await pipeline.vllm_server_ready(args)
    return task, semaphore


async def check_external_server_ready(args: SimpleNamespace) -> None:
    """Wait for an external server to be ready using the pipeline's readiness probe."""
    # For external server, BASE_SERVER_PORT is irrelevant, but readiness needs server/api_key in args
    await pipeline.vllm_server_ready(args)


async def process_pdf_bytes(args: SimpleNamespace, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Process one PDF or image file provided as bytes and return a Dolma-style document.

    Persists nothing; callers decide how to persist single vs batch writes.
    """
    # Write incoming bytes to a temp file path so we can reuse pipeline utilities
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as tf:
        tf.write(file_bytes)
        tf.flush()
        bin_path = tf.name

    local_pdf_path = None
    try:
        # Convert images to PDF if needed
        if is_png(bin_path) or is_jpeg(bin_path):
            pdf_bytes = convert_image_to_pdf_bytes(bin_path)
            with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as pdf_tf:
                pdf_tf.write(pdf_bytes)
                pdf_tf.flush()
                local_pdf_path = pdf_tf.name
        else:
            # Already a PDF; reuse the written path
            local_pdf_path = bin_path

        # Count pages
        try:
            reader = PdfReader(local_pdf_path)
            num_pages = reader.get_num_pages()
        except Exception:
            pipeline.logger.exception(f"Could not count number of pages for {filename}, aborting document")
            return None

        # Process all pages concurrently using pipeline.process_page
        page_tasks: List[asyncio.Task] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for page_num in range(1, num_pages + 1):
                    task = tg.create_task(pipeline.process_page(args, 0, filename, local_pdf_path, page_num))
                    page_tasks.append(task)

            page_results = [t.result() for t in page_tasks]

            num_fallback_pages = sum(p.is_fallback for p in page_results)
            if num_pages > 0 and num_fallback_pages / num_pages > args.max_page_error_rate:
                pipeline.logger.error(
                    f"Document {filename} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
                )
                return None

            return pipeline.build_dolma_document(filename, page_results)
        except Exception as ex:
            pipeline.logger.exception(f"Exception in process_pdf_bytes for {filename}: {ex}")
            return None
    finally:
        # Remove the original uploaded temp if we didn't already set local_pdf_path to it
        if bin_path and local_pdf_path != bin_path:
            try:
                if os.path.exists(bin_path):
                    os.unlink(bin_path)
            except Exception:
                pass
        if local_pdf_path:
            try:
                if os.path.exists(local_pdf_path):
                    os.unlink(local_pdf_path)
            except Exception:
                pass


def ensure_workspace_dirs(workspace: str) -> Tuple[str, str]:
    results_dir = os.path.join(workspace, "results")
    markdown_dir = os.path.join(workspace, "markdown")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)
    return results_dir, markdown_dir


def persist_single_doc(doc: Dict[str, Any], workspace: str, upload_basename: str, write_markdown: bool) -> str:
    results_dir, markdown_dir = ensure_workspace_dirs(workspace)
    out_uuid = uuid.uuid4().hex
    jsonl_path = os.path.join(results_dir, f"output_{out_uuid}.jsonl")
    # Write one line JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        import json as _json

        f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    if write_markdown:
        md_path = os.path.join(markdown_dir, f"{os.path.splitext(os.path.basename(upload_basename))[0]}.md")
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(doc.get("text", ""))
    return jsonl_path


def persist_batch_docs(docs: List[Dict[str, Any]], workspace: str, upload_basenames: List[str], write_markdown: bool) -> str:
    results_dir, markdown_dir = ensure_workspace_dirs(workspace)
    out_uuid = uuid.uuid4().hex
    jsonl_path = os.path.join(results_dir, f"output_{out_uuid}.jsonl")
    # Write many line JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        import json as _json

        for d in docs:
            f.write(_json.dumps(d, ensure_ascii=False) + "\n")

    if write_markdown:
        for name, d in zip(upload_basenames, docs):
            md_path = os.path.join(markdown_dir, f"{os.path.splitext(os.path.basename(name))[0]}.md")
            with open(md_path, "w", encoding="utf-8") as mf:
                mf.write(d.get("text", ""))
    return jsonl_path
