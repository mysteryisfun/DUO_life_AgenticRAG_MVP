import uuid
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from models import QueryRequest, ClearMemoryRequest, SessionResponse, ClearMemoryResponse, TokenChunk, FinalResponseData
from advanced_rag_agent import get_agent_executor, get_chat_history, store as agent_session_store

# --- App Initialization ---
app = FastAPI(
    title="DuoLife RAG Agent API",
    description="API for interacting with the advanced RAG agent for DuoLife products and business.",
    version="1.0.0"
)

# --- Agent Initialization ---
agent_executor = get_agent_executor()

# --- API Endpoints ---

@app.get("/new_session", response_model=SessionResponse)
async def new_session():
    """
    Generates a new session ID to start a new conversation.
    """
    session_id = str(uuid.uuid4())
    get_chat_history(session_id)
    return SessionResponse(session_id=session_id)

@app.post("/query")
async def query(request: QueryRequest) -> Dict[str, Any]:
    """
    Receives a question and session_id, invokes the agent, and returns a single response.
    """
    session_id = request.session_id
    question = request.question

    if session_id not in agent_session_store:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    try:
        # 1. Invoke the agent and wait for the final result
        full_response_obj = await agent_executor.ainvoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

        # 2. Parse the final answer from the agent's response
        final_answer_data = full_response_obj.get("final_answer", {})
        conversational_part = final_answer_data.get("conversational_answer", "No answer generated.")
        sources_data = final_answer_data.get("sources", [])

        # 3. Construct the single JSON response
        response_data = {
            "session_id": session_id,
            "final_answer": conversational_part,
            "sources": sources_data
        }
        
        # Return the final object with type "token"
        return {"type": "token", "data": response_data}

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


@app.post("/clear_memory", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest):
    """
    Clears the conversation memory for a given session.
    """
    session_id = request.session_id
    if session_id in agent_session_store:
        agent_session_store[session_id].clear()
        return ClearMemoryResponse(message=f"Conversation memory for session {session_id} has been cleared.")
    else:
        raise HTTPException(status_code=404, detail="Session not found.")