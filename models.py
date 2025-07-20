from pydantic import BaseModel, Field
from typing import List, Optional

# --- Request Models ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question for the agent.")
    session_id: str = Field(..., description="The unique ID for the conversation session.")

class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., description="The session ID to be cleared.")

# --- Response Models ---

class SessionResponse(BaseModel):
    session_id: str

class ClearMemoryResponse(BaseModel):
    message: str

# --- Streaming Response Models (for documentation, not direct return types) ---

class LogData(BaseModel):
    message: str

class LogMessage(BaseModel):
    type: str = "log"
    data: LogData

class TokenData(BaseModel):
    chunk: str

class TokenChunk(BaseModel):
    type: str = "token"
    data: TokenData

class Source(BaseModel):
    name: str
    links: List[str]
    type: str
    category: str
    snippet: str

class FinalResponseData(BaseModel):
    session_id: str
    final_answer: str
    sources: List[Source]

class FinalResponse(BaseModel):
    type: str = "final_response"
    data: FinalResponseData