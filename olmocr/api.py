import asyncio
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from olmocr.api_service import (
    check_external_server_ready,
    load_args_from_env,
    persist_batch_docs,
    persist_single_doc,
    process_pdf_bytes,
    start_local_vllm_and_wait,
)

# Globals to manage lifecycle
_startup_lock = asyncio.Lock()
_local_task: asyncio.Task | None = None
_semaphore: asyncio.Semaphore | None = None
_ready_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _local_task, _semaphore
    # Startup
    async with _startup_lock:
        if not _ready_event.is_set():
            args = load_args_from_env()
            if args.server:
                await check_external_server_ready(args)
                _ready_event.set()
            else:
                task, semaphore = await start_local_vllm_and_wait(args)
                _local_task = task
                _semaphore = semaphore
                _ready_event.set()

    yield

    # Shutdown
    if _local_task is not None:
        _local_task.cancel()
        try:
            await _local_task
        except Exception:
            pass
        _local_task = None


app = FastAPI(title="olmOCR API", lifespan=lifespan)


@app.get("/health")
async def health():
    if _ready_event.is_set():
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "initializing"})


@app.post("/process")
async def process(file: UploadFile = File(...)):
    if not _ready_event.is_set():
        raise HTTPException(status_code=503, detail="Initializing")

    args = load_args_from_env()
    data = await file.read()
    filename = file.filename or "upload.bin"
    doc = await process_pdf_bytes(args, data, filename)
    if doc is None:
        raise HTTPException(status_code=400, detail="Failed to process document")

    # Persist outputs
    try:
        persist_single_doc(doc, args.workspace, filename, args.markdown)
    except Exception:
        # Persistence errors should not break API response, but log if needed
        pass

    return JSONResponse(content=doc)


@app.post("/process_batch")
async def process_batch(files: List[UploadFile] = File(...)):
    if not _ready_event.is_set():
        raise HTTPException(status_code=503, detail="Initializing")

    args = load_args_from_env()
    # Read all files first
    filenames = [(f.filename or "upload.bin") for f in files]
    file_datas = [await f.read() for f in files]

    results = []
    docs = []
    success_names: List[str] = []
    for name, data in zip(filenames, file_datas):
        try:
            doc = await process_pdf_bytes(args, data, name)
            if doc is None:
                results.append({"filename": name, "error": "failed"})
            else:
                results.append(doc)
                docs.append(doc)
                success_names.append(name)
        except Exception as ex:
            results.append({"filename": name, "error": str(ex)})

    # Persist batch JSONL and optional markdown for successful docs only
    try:
        if docs:
            persist_batch_docs(docs, args.workspace, success_names, args.markdown)
    except Exception:
        pass

    return JSONResponse(content=results)
