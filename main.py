import uuid
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from models import QueryRequest, ClearMemoryRequest, SessionResponse, ClearMemoryResponse
from advanced_rag_agent import get_agent_executor, get_chat_history, store as agent_session_store
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
# --- App Initialization ---
app = FastAPI(
    title="DuoLife RAG Agent API",
    description="API for interacting with the advanced RAG agent for DuoLife products and business.",
    version="1.0.0"
)
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

@app.get("/")
async def serve_react():
    return FileResponse("frontend/build/index.html")
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Agent Initialization ---
agent_executor = get_agent_executor()

# --- API Endpoints ---

@app.get("/new_session", response_model=SessionResponse)
async def new_session():
    """Generates a new session ID to start a new conversation."""
    session_id = str(uuid.uuid4())
    get_chat_history(session_id)
    return SessionResponse(session_id=session_id)

@app.post("/query")
async def query(request: QueryRequest) -> Dict[str, Any]:
    """Receives a question and session_id, invokes the agent, and returns the response."""
    session_id = request.session_id
    question = request.question

    if session_id not in agent_session_store:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    try:
        # Invoke the agent and get the final state
        response_state = await agent_executor.ainvoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

        # Extract the final answer directly
        final_answer_data = response_state.get("final_answer", {})
        conversational_part = final_answer_data.get("conversational_answer", "No answer generated.")
        sources_data = final_answer_data.get("sources", [])

        # Construct the response payload
        response_payload = {
            "session_id": session_id,
            "final_answer": conversational_part,
            "sources": sources_data
        }
        
        return {"type": "final_response", "data": response_payload}

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.post("/clear_memory", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest):
    """Clears the conversation memory for a given session."""
    session_id = request.session_id
    if session_id in agent_session_store:
        agent_session_store[session_id].clear()
        return ClearMemoryResponse(message=f"Conversation memory for session {session_id} has been cleared.")
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

