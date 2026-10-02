"""
SalahSetu AI Service - FastAPI Main Application Entry Point
"""

import sys
import os
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
base_dir = Path(__file__).resolve().parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Try loading python-dotenv if present
try:
    from dotenv import load_dotenv
    load_dotenv(base_dir / "ai_service" / ".env")
except ImportError:
    pass

from ai_service.api.schemas import (
    LegalQueryRequest,
    LegalQueryResponse,
    HealthResponse,
    ReadinessResponse
)
from ai_service.legal_query_engine import LegalQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SalahSetuAIService")

app = FastAPI(
    title="SalahSetu AI Service",
    description="Official-Source Citizen Legal Intelligence API for Bharatiya Nyaya Sanhita (BNS)",
    version="1.0.0"
)

# Configure CORS from environment variable
cors_origins_raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
allow_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Lazy singleton engine instance
_engine_instance: LegalQueryEngine = None


def get_engine() -> LegalQueryEngine:
    global _engine_instance
    if _engine_instance is None:
        try:
            _engine_instance = LegalQueryEngine()
        except Exception as e:
            logger.error(f"Failed to initialize LegalQueryEngine: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI Query Engine artifacts not ready."
            )
    return _engine_instance


# Exception Handlers to ensure safe error outputs without leaking stack traces or secrets
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input format",
            "message": "Question must be a non-empty string under 1000 characters."
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": str(exc)
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error during request processing: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An error occurred while processing the legal query. Please try again later."
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness check confirming the API server process is running."""
    return HealthResponse(
        status="ok",
        service="SalahSetu AI Service"
    )


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """Readiness check confirming required local AI artifacts can be loaded."""
    base_path = Path(__file__).resolve().parent.parent.parent
    faiss_path = base_path / "data" / "embeddings" / "BNS_faiss.index"
    metadata_path = base_path / "data" / "embeddings" / "BNS_embedding_metadata.json"
    chunks_path = base_path / "data" / "chunks" / "BNS_chunks.json"
    sections_path = base_path / "data" / "structured" / "BNS_sections.json"

    details = {
        "faiss_index_exists": faiss_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "chunks_exists": chunks_path.exists(),
        "sections_exists": sections_path.exists()
    }

    all_ready = all(details.values())

    if not all_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "artifacts_ready": False,
                "details": details
            }
        )

    return ReadinessResponse(
        status="ready",
        artifacts_ready=True,
        details=details
    )


@app.post("/api/v1/legal/query", response_model=LegalQueryResponse, tags=["Legal Query"])
async def query_legal_engine(request_data: LegalQueryRequest):
    """
    Executes grounded legal query pipeline against Bharatiya Nyaya Sanhita (BNS) dataset.
    """
    engine = get_engine()

    try:
        raw_res = engine.ask(request_data.question)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error executing engine query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing legal query."
        )

    response_data = {
        "question": raw_res["question"],
        "answer": raw_res["answer"],
        "sources": raw_res["sources"],
        "limitations": raw_res["limitations"],
        "retrieved_sections": raw_res["retrieved_sections"]
    }

    return response_data
