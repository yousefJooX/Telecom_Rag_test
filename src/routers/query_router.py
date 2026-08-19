from fastapi import APIRouter, HTTPException
import logging
from src.models.schemas import QueryRequest, QueryResponse
from src.services.rag_service import RAGService
from src.logging.logger import logger

router = APIRouter(prefix="/api/v1", tags=["Query"])
rag_service = RAGService()


@router.post("/query", response_model=QueryResponse, status_code=200)
async def answer_ticket(payload: QueryRequest):
    ticket = payload.ticket.strip()
    if not ticket:
        raise HTTPException(400, "Ticket text cannot be empty.")
    try:
        return rag_service.answer_ticket(ticket)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(500, "No index found. Ingest a document first via /api/v1/ingest.")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, "Internal server error.")