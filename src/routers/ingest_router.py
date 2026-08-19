from fastapi import APIRouter, HTTPException, UploadFile, File
from src.models.schemas import IngestResponse
from src.services.ingest_service import IngestionService
from src.logging.logger import logger
from src.config.config_parser import settings

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])
ingestion_service = IngestionService()


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def upload_and_ingest(file: UploadFile = File(..., description="Select a .txt, .md, or .csv file to ingest")):
    if not file.filename.lower().endswith((".txt", ".md", ".csv")):
        raise HTTPException(400, "Only .txt, .md, .csv files are supported.")

    try:
        content = await file.read()
        chunks = ingestion_service.process_file(content, file.filename)
        return IngestResponse(
            message="File ingested and indexed successfully.",
            chunks_indexed=chunks,
            index_path=settings.vector_index_path
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, "Internal error during ingestion.")