import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import get_chatbot_response, warm_up
from lead_capture import (
    process_lead_message,
    get_conversation,
    offer_followup,
    save_conversation,
    pending_prompt
)

warm_up()

# Last few turns per conversation (in memory) so follow-ups work.
HISTORY = {}


def remember(conversation_id, question, answer):
    turns = HISTORY.setdefault(conversation_id, [])
    turns.append({"role": "user", "content": question})
    turns.append({"role": "assistant", "content": answer})
    del turns[:-12]


def is_unknown_answer(answer):
    # Any "not verified" answer (fixed fallback or a careful hedge) offers
    # a human follow-up.
    return "don't have verified" in answer.lower()


def conversation_is_active(conversation_id):
    conv = get_conversation(conversation_id)
    return bool(conv and not conv["completed"])


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

        if re.search(r"start (the |a )?(conversation|chat) (again|over)|"
                     r"start over|reset (the )?(chat|conversation)|new conversation",
                     question.lower()):
            HISTORY.pop(conversation_id, None)
            save_conversation(conversation_id, "reset", {}, "cancelled", 1)
            return ChatResponse(
                answer="Sure, let's start fresh. How can I help you with Leadonspot?"
            )

        # STEP 1: lead capture (demo / contact request / active form)
        lead_answer, handled = process_lead_message(
            conversation_id, question
        )

        if handled:
            remember(conversation_id, question, lead_answer)
            return ChatResponse(answer=lead_answer)

        # STEP 2: normal RAG answer (with chat history for follow-ups)
        history = HISTORY.get(conversation_id, [])

        answer = get_chatbot_response(question, history)

        # If the user was mid-form and asked a side question,
        # answer it and then re-ask the pending field.
        pending = pending_prompt(conversation_id)

        if pending:
            answer = f"{answer}\n\n{pending}"

        # STEP 3: bot could not answer -> offer human follow-up
        elif is_unknown_answer(answer) and not conversation_is_active(conversation_id):
            answer = f"{answer}\n\n{offer_followup(conversation_id)}"

        remember(conversation_id, question, answer)

        return ChatResponse(answer=answer)

    except Exception as e:

        print("Chatbot error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )