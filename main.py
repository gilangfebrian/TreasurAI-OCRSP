import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from extractor import extract_invoice_data, extract_invoice_trai
from models import ExtractionError, InvoiceResponse
from preprocessor import MAX_PAGES, preprocess_file, extract_raw_text

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    yield
    app.state.openai.close()


app = FastAPI(
    title="TreasurAI Invoice Extractor API",
    description="Extract structured data from invoice images and PDFs.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.post("/extract-invoice", response_model=InvoiceResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    try:
        images_b64, total_pages = preprocess_file(raw_bytes, file.content_type)
    except Exception as e:
        logger.exception("Failed to preprocess file: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not process file: {e}")

    truncated = file.content_type == "application/pdf" and total_pages > MAX_PAGES

    try:
        result = extract_invoice_data(app.state.openai, images_b64)
    except OpenAIError as e:
        logger.exception("OpenAI API error: %s", e)
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")
    except (ValueError, ValidationError) as e:
        logger.exception("Extraction/validation error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during extraction: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during extraction.")

    response = result.model_dump()
    if truncated:
        response["_warning"] = (
            f"PDF has {total_pages} pages; only the first {MAX_PAGES} were processed."
        )

    return JSONResponse(content=response)


@app.post("/extract-invoice-trai", response_model=InvoiceResponse)
async def extract_invoice_trai_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    try:
        ocr_text, total_pages = extract_raw_text(raw_bytes, file.content_type)
    except Exception as e:
        logger.exception("OCR preprocessing failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not process file: {e}")

    if not ocr_text.strip():
        raise HTTPException(
            status_code=422,
            detail="OCR tidak menghasilkan teks apapun. Pastikan file berisi teks yang terbaca.",
        )

    truncated = file.content_type == "application/pdf" and total_pages > MAX_PAGES

    try:
        result = extract_invoice_trai(ocr_text)
    except ValueError as e:
        logger.exception("TRAI extraction error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected TRAI error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during extraction.")

    response = result.model_dump()
    if truncated:
        response["_warning"] = (
            f"PDF has {total_pages} pages; only the first {MAX_PAGES} were processed."
        )

    return JSONResponse(content=response)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ExtractionError(detail="Unexpected server error.").model_dump(),
    )