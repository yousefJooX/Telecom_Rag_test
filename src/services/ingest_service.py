from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore.database import VectorDatabaseRepository
from src.logging.logger import logger
 
class IngestionService:
    """
    Business Logic Layer: Ingestion Service.
    Handles processing dynamic uploaded files from FastAPI endpoints into semantic vector chunks.
    """
    
    def __init__(self):    
        self.repo = VectorDatabaseRepository()

    async def process_uploaded_file(
        self, 
        file: UploadFile, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100
    ) -> int:
        logger.info(f"Reading uploaded file content for: {file.filename}")
        content = await file.read()
        text_content = content.decode("utf-8").replace("\r\n", "\n")

        logger.info(f"Splitting uploaded file '{file.filename}' (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\r\n\r\n", "\n", "\r\n", " ", ""]
        )
        
        # Create documents directly from memory with source metadata
        docs = text_splitter.create_documents(
            texts=[text_content],
            metadatas=[{"source": file.filename}]
        )
        logger.info(f"Generated {len(docs)} text chunks from '{file.filename}'.")

        self.repo.create_from_documents(docs)
        return len(docs)