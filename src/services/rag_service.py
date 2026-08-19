import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.vectorstore.database import VectorDatabaseRepository
from src.services.ingest_service import IngestionService
from src.models.schemas import QueryResponse
from src.logging.logger import logger
from src.config.config_parser import settings


class RAGService:
    """
    RAG Service: Retrieval-Augmented Generation for ticket answering.
    """

    def __init__(self):
        from src.core.factories import ModelFactory
        self.repo = VectorDatabaseRepository()
        self.llm = ModelFactory.get_llm()
        self._prompt = self._build_prompt()

    def _format_docs(self, docs) -> str:
        """Join retrieved document page contents with double newlines."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the prompt template with Egyptian-Arabic customer-support persona."""
        return ChatPromptTemplate.from_template(
            """You are a Telecom customer support assistant. Use the provided context to answer the customer's ticket.

Guidelines:
- Answer in the language the ticket is written (Arabic or English).
- Do not mention real telecom brand names; use generic terms only.
- Do not use data outside the injected context.
- If the issue requires a technician visit, escalate appropriately.
- If the answer is not fully contained in the context, say you don't have enough information.

Context:
{context}

Ticket:
{ticket}

Response:"""
        )

    def answer_ticket(self, ticket: str) -> dict:
        """
        Answer a customer ticket using RAG.

        Returns dict matching QueryResponse schema.
        """
        start = time.time()

        try:
            vectorstore = self.repo.load_index()
        except FileNotFoundError:
            raise FileNotFoundError("No index found. Ingest a document first.")

        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.k_retrieval})

        chain = (
            {"context": retriever | self._format_docs, "ticket": RunnablePassthrough()}
            | self._prompt
            | self.llm
            | StrOutputParser()
        )

        response_text = chain.invoke(ticket)

        # Capture token usage from LLM response metadata
        usage = getattr(self.llm, "response_metadata", {}).get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        execution_time = time.time() - start

        return {
            "response": response_text,
            "sources_count": settings.k_retrieval,
            "execution_time": execution_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }