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

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://duolife-2170f667c572.herokuapp.com",
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

# --- API Endpoints (MUST COME BEFORE STATIC MOUNT) ---

@app.get("/new_session", response_model=SessionResponse)
async def new_session():
    """Generates a new session ID to start a new conversation."""
    session_id = str(uuid.uuid4())
    # Initialize the session - let get_chat_history handle the proper initialization
    get_chat_history(session_id)
    return SessionResponse(session_id=session_id)

@app.post("/query")
async def query(request: QueryRequest) -> Dict[str, Any]:
    """Receives a question and session_id, invokes the agent, and returns the response."""
    session_id = request.session_id
    question = request.question

    # Check if session exists by trying to get chat history
    try:
        chat_history = get_chat_history(session_id)
        if chat_history is None:
            raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")
    except:
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
    try:
        chat_history = get_chat_history(session_id)
        if chat_history and hasattr(chat_history, 'clear'):
            chat_history.clear()
            return ClearMemoryResponse(message=f"Conversation memory for session {session_id} has been cleared.")
        else:
            raise HTTPException(status_code=404, detail="Session not found.")
    except:
        raise HTTPException(status_code=404, detail="Session not found.")

# --- Static Files Mount (MUST BE LAST) ---
# This will serve your React app for any route not matched above
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")