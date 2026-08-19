from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore.database import VectorDatabaseRepository
from src.logging.logger import logger
from src.config.config_parser import settings


class IngestionService:
    """
    Business Logic Layer: Ingestion Service.
    Handles processing uploaded file content into semantic vector chunks.
    """

    def __init__(self):
        self.repo = VectorDatabaseRepository()

    def process_file(self, content: str, filename: str, chunk_size=None, chunk_overlap=None) -> int:
        """
        Process file content into chunks and index into FAISS.
        Validates extension caller-side (in router); this method assumes valid extension.
        """
        logger.info(f"Processing file content for: {filename}")

        # Strip Windows line endings
        text_content = content.replace("\r\n", "\n")

        logger.info(f"Splitting file '{filename}' (chunk_size={chunk_size or settings.chunk_size}, overlap={chunk_overlap or settings.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\r\n\r\n", "\n", "\r\n", " ", ""]
        )

        # Create documents directly from memory with source metadata
        docs = text_splitter.create_documents(
            texts=[text_content],
            metadatas=[{"source": filename}]
        )
        logger.info(f"Generated {len(docs)} text chunks from '{filename}'.")

        self.repo.create_from_documents(docs)
        return len(docs)