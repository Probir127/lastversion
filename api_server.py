import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import backend
import os

app = FastAPI(
    title="HR Chatbot API",
    description="REST API for the RAG-powered HR Chatbot using Gemini AI."
)

# Add CORS middleware for web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list = []
    last_person: str | None = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        ai_response = backend.get_response(
            user_input=request.user_input,
            chat_history=request.chat_history,
            last_person=request.last_person
        )
        return {"response": ai_response}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")