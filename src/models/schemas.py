from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int = Field(..., gt=0, description="Number of chunks indexed")
    index_path: str


class QueryRequest(BaseModel):
    ticket: str = Field(..., min_length=1, description="Customer complaint/ticket text")


class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated AI response")
    sources_count: int = Field(..., ge=0, description="Number of context chunks retrieved")
    execution_time: float = Field(..., description="Execution time in seconds")
    prompt_tokens: int = Field(..., description="Prompt tokens sent to LLM")
    completion_tokens: int = Field(..., description="Completion tokens from LLM")
    total_tokens: int = Field(..., description="Total tokens used")