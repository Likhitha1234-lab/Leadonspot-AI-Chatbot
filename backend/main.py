
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import get_chatbot_response
from lead_capture import (
    process_lead_message,
    get_conversation
)


# --------------------------------------------------
# 1. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Leadonspot AI Chatbot",
    description="AI-powered chatbot for Leadonspot",
    version="1.0"
)


# --------------------------------------------------
# 2. CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. REQUEST AND RESPONSE MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    conversation_id: str


class ChatResponse(BaseModel):
    answer: str


# --------------------------------------------------
# 4. HOME ROUTE
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Leadonspot Chatbot API is running!"
    }


# --------------------------------------------------
# 5. CHAT ROUTE
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    question = request.question.strip()
    conversation_id = request.conversation_id.strip()

    if not question:
        return ChatResponse(
            answer="Please enter a message."
        )

    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail="Conversation ID is missing."
        )

    print("\n================================")
    print("NEW CHAT MESSAGE")
    print("Conversation ID:", conversation_id)
    print("Question:", question)
    print("================================")

    try:

        # ------------------------------------------
        # STEP 1: CHECK EXISTING LEAD CONVERSATION
        # ------------------------------------------

        conversation = get_conversation(conversation_id)

        if conversation and not conversation["completed"]:

            print("Active lead conversation found.")
            print("Current step:", conversation["current_step"])

            lead_answer, handled = process_lead_message(
                conversation_id,
                question
            )

            if handled:

                print("Lead capture handled the message.")

                return ChatResponse(
                    answer=lead_answer
                )

        # ------------------------------------------
        # STEP 2: CHECK FOR NEW SALES OR DEMO INTENT
        # ------------------------------------------

        else:

            lead_answer, handled = process_lead_message(
                conversation_id,
                question
            )

            if handled:

                print("New lead capture started.")

                return ChatResponse(
                    answer=lead_answer
                )

        # ------------------------------------------
        # STEP 3: USE NORMAL RAG CHATBOT
        # ------------------------------------------

        print("No active lead capture. Using RAG.")

        answer = get_chatbot_response(question)

        return ChatResponse(
            answer=answer
        )

    except Exception as e:

        print("Chatbot error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )
