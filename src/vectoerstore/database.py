import os
from langchain_community.vectorstores import FAISS
from src.core.factories import ModelFactory
from src.config.config_parser import settings
from src.logging.logger import logger

class VectorDatabaseRepository:
    """
    Repository Layer managing vector database operations (FAISS).
    Uses in-memory Singleton caching so disk I/O only happens ONCE.
    """
    _instance_cache = None  # Class-level RAM cache for FAISS

    def __init__(self):
        self.embeddings = ModelFactory.get_embeddings()
        self.index_path = settings.vector_index_path

    def save_index(self, vectorstore: FAISS):
        logger.info(f"Saving FAISS index locally to '{self.index_path}'...")
        vectorstore.save_local(self.index_path)
        # Update RAM cache with the new vectorstore
        VectorDatabaseRepository._instance_cache = vectorstore

    def load_index(self) -> FAISS:
        # 1. If vectorstore is already loaded in RAM, return immediately! (Zero Disk I/O)
        if VectorDatabaseRepository._instance_cache is not None:
            logger.info("⚡ Using in-memory cached FAISS index (0ms disk latency).")
            return VectorDatabaseRepository._instance_cache

        # 2. Otherwise, load from disk for the first time
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index folder '{self.index_path}' not found! Upload a file via /api/v1/ingest first."
            )
            
        logger.info(f"Loading FAISS index from disk '{self.index_path}' into memory...")
        vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        # Cache in memory for subsequent requests
        VectorDatabaseRepository._instance_cache = vectorstore
        return vectorstore

    def create_from_documents(self, documents: list, batch_size: int = None) -> FAISS:
        batch_size = batch_size or settings.batch_size
        total_chunks = len(documents)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        logger.info(f"Indexing {total_chunks} chunks into FAISS in {total_batches} batches (batch_size={batch_size})...")
        vectorstore = None
        
        for i in range(0, total_chunks, batch_size):
            batch = documents[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            logger.info(f"Embedding & indexing batch {current_batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.add_documents(batch)

        self.save_index(vectorstore)
        return vectorstore