
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import get_chatbot_response


# Initialize FastAPI
app = FastAPI(
    title="Leadonspot AI Chatbot",
    description="AI-powered chatbot for Leadonspot",
    version="1.0"
)


# Allow Next.js frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class ChatRequest(BaseModel):
    question: str


# Response model
class ChatResponse(BaseModel):
    answer: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Leadonspot Chatbot API is running!"
    }


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = get_chatbot_response(request.question)

    return {
        "answer": answer
    }
