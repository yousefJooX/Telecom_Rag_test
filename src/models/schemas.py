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


class StatsResponse(BaseModel):
    total_queries: int = Field(0, ge=0, description="Total number of queries processed")
    total_prompt_tokens: int = Field(0, ge=0, description="Total prompt tokens across all queries")
    total_completion_tokens: int = Field(0, ge=0, description="Total completion tokens across all queries")
    total_tokens: int = Field(0, ge=0, description="Total tokens across all queries")
    total_cost_usd: float = Field(0.0, ge=0, description="Total cost in USD across all queries")
    avg_execution_time: float = Field(0.0, ge=0, description="Average execution time in seconds")