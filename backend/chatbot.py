
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "leadonspot_kb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Load API key
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


# Connect to ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)


# Check database
total_records = vectorstore._collection.count()

print("Total records:", total_records)

if total_records == 0:
    raise ValueError(
        "ChromaDB is empty! Run python ingest.py first."
    )


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


def get_chatbot_response(question):

    # 1. Retrieve relevant documents
    documents = retriever.invoke(question)

    if not documents:
        return (
            "I couldn't find that information right now. "
            "Please contact the Leadonspot team for more details."
        )

    # 2. Prepare context
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # 3. Create prompt
    prompt = f"""
You are Leadonspot's AI assistant.

Your role is to help users understand Leadonspot
and its products through natural conversations.

Use the following context to answer the user's question.

Context:
{context}

User Question:
{question}

Instructions:

- Talk like a friendly person having a normal conversation.
- Use simple, everyday English.

For YES/NO questions:
- Start with a clear "Yes" or "No" whenever appropriate.
- Give only one short sentence explaining the answer.
- Avoid unnecessary details.

For descriptive questions:
- Keep the answer to a maximum of 2 sentences.
- Mention only the most important information.
- Do not explain every feature unless the user asks.

General rules:
- Avoid formal, robotic, or repetitive language.
- Do not use unnecessary headings or bullet points.
- Answer directly without lengthy introductions.
- Do not repeat the user's question.
- Do not invent features, prices, or company information.
- Use only the information available in the context.
- Do not combine features of different products.
- Never assume integrations, pricing, or capabilities.
- If information is unavailable, politely say so.
- For follow-up questions, answer only what the user asks.
- Prioritize short and accurate answers over lengthy explanations.

Answer:
"""

    # 4. Generate answer using Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        reasoning_effort="low",
        include_reasoning=False
    )

    # 5. Handle AI response safely
    answer = response.choices[0].message.content

    # Debug information
    print("Finish reason:", response.choices[0].finish_reason)
    print("AI response:", answer)

    if answer and answer.strip():
        return answer.strip()

    return (
        "Sorry, I couldn't generate a response. "
        "Please try again."
    )


# Terminal testing
if __name__ == "__main__":

    print("\nLeadonspot Chatbot Ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Chatbot stopped.")
            break

        if not question:
            continue

        try:
            answer = get_chatbot_response(question)
            print("\nBot:", answer, "\n")

        except Exception as e:
            print("Error:", str(e))
