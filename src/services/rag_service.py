import time
from langchain_core.prompts import ChatPromptTemplate
from src.vectorstore.database import VectorDatabaseRepository
from src.logging.logger import logger
from src.config.config_parser import settings


# Simple in-memory counter: list of per-request dicts (appended on each query)
# In a production system this would be a DB, but for MVP it's a module-level list.
_query_token_history: list[dict] = []


def aggregate_token_stats() -> dict:
    """Return aggregated token stats and total cost from the in-memory query history."""
    if not _query_token_history:
        return {"total_queries": 0}
    total = {
        "total_queries": len(_query_token_history),
        "total_prompt_tokens": sum(q["prompt_tokens"] for q in _query_token_history),
        "total_completion_tokens": sum(q["completion_tokens"] for q in _query_token_history),
        "total_tokens": sum(q["total_tokens"] for q in _query_token_history),
        "total_cost_usd": sum(q["cost_usd"] for q in _query_token_history),
        "avg_execution_time": sum(q["execution_time"] for q in _query_token_history) / len(_query_token_history),
    }
    return total


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
        retrieved_docs = retriever.invoke(ticket)
        context = self._format_docs(retrieved_docs)

        chain = self._prompt | self.llm

        ai_message = chain.invoke({"context": context, "ticket": ticket})
        response_text = ai_message.content

        usage_meta = getattr(ai_message, "usage_metadata", {}) or {}
        if usage_meta:
            prompt_tokens = usage_meta.get("input_tokens", 0)
            completion_tokens = usage_meta.get("output_tokens", 0)
            total_tokens = usage_meta.get("total_tokens", 0)
        else:
            usage = getattr(ai_message, "response_metadata", {}).get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

        execution_time = time.time() - start

        # --- Cost calculation (Gemini 1.5 Flash pricing example) ---
        prompt_cost = (prompt_tokens / 1_000_000) * settings.cost_per_1m_input_tokens
        completion_cost = (completion_tokens / 1_000_000) * settings.cost_per_1m_output_tokens
        total_cost = prompt_cost + completion_cost

        # Record token usage for aggregation
        _query_token_history.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": total_cost,
            "execution_time": execution_time,
        })

        # Log per-request cost (INFO level; aggregation handled in Part 2-below)
        logger.info(
            f"Query cost — prompt: {prompt_tokens} tok (${prompt_cost:.6f}), "
            f"completion: {completion_tokens} tok (${completion_cost:.6f}), "
            f"total: {total_tokens} tok (${total_cost:.6f})"
        )

        return {
            "response": response_text,
            "sources_count": len(retrieved_docs),
            "execution_time": execution_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }