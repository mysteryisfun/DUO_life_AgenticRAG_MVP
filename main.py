import uuid
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import Dict, AsyncGenerator
import json

from models import QueryRequest, ClearMemoryRequest, SessionResponse, ClearMemoryResponse
from advanced_rag_agent import get_agent_executor, get_chat_history
from langchain_core.messages import AIMessage, HumanMessage

# --- App Initialization ---
app = FastAPI(
    title="DuoLife RAG Agent API",
    description="API for interacting with the advanced RAG agent for DuoLife products and business.",
    version="1.0.0"
)

# --- In-Memory Session Storage ---
# Note: For production, consider a more persistent storage like Redis.
SESSIONS: Dict[str, any] = {}

# --- Agent Initialization ---
agent_executor = get_agent_executor()

# --- API Endpoints ---

@app.get("/new_session", response_model=SessionResponse)
async def new_session():
    """
    Generates a new session ID to start a new conversation.
    """
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = get_chat_history() # Initialize chat history
    return SessionResponse(session_id=session_id)

@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """
    Receives a question and session_id, streams the agent's response.
    """
    session_id = request.session_id
    question = request.question

    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    chat_history = SESSIONS[session_id]

    // ...existing code...
    async def stream_generator() -> AsyncGenerator[str, None]:
        # 1. Log that the process is starting
        log_msg = {"type": "log", "data": {"message": "Processing query..."}}
        yield f"data: {json.dumps(log_msg)}\n\n"

        full_response_obj = {}
        # 2. Stream the agent's response
        async for chunk in agent_executor.astream(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        ):
            # We are interested in the final output from the agent, which is under the key '__end__'
            if "__end__" in chunk:
                full_response_obj = chunk["__end__"]['output']
                break # End the loop once we have the final response
        
        # 3. Parse the final response from the agent
        final_answer_str = full_response_obj.get("answer", "")
        conversational_part = ""
        sources_json = []

        # Split the final answer string by our separator
        separator = "---JSON_SOURCES---"
        if separator in final_answer_str:
            parts = final_answer_str.split(separator)
            conversational_part = parts[0].strip()
            try:
                sources_json = json.loads(parts[1].strip())
            except json.JSONDecodeError:
                print("Error decoding sources JSON from agent response.")
                sources_json = [] # Default to empty list on error
        else:
            conversational_part = final_answer_str.strip()

        # 4. Yield the conversational part as tokens (simulating token streaming)
        token_msg = {"type": "token", "data": {"chunk": conversational_part}}
        yield f"data: {json.dumps(token_msg)}\n\n"

        # 5. Send the final structured response
        final_response = {
            "type": "final_response",
            "data": {
                "session_id": session_id,
                "final_answer": conversational_part,
                "sources": sources_json
            }
        }
        yield f"data: {json.dumps(final_response)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/clear_memory", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest):
    """
    Clears the conversation memory for a given session.
    """
    session_id = request.session_id
    if session_id in SESSIONS:
        SESSIONS[session_id].clear()
        return ClearMemoryResponse(message=f"Conversation memory for session {session_id} has been cleared.")
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

# To run this app:
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi "uvicorn[standard]"
# 2. Run the server from your terminal:
#    uvicorn main:app --reload