
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DB_PATH = BASE_DIR / "chroma_db"
KB_PATH = PROJECT_DIR / "knowledge_base"
PRICING_FILE = KB_PATH / "pricing.txt"

COLLECTION_NAME = "leadonspot_kb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Starting relevance threshold. Tune after testing.
RELEVANCE_THRESHOLD = 0.25

UNKNOWN_ANSWER = (
    "I don't have verified information about that. "
    "Please contact the Leadonspot team for confirmation."
)

OUT_OF_SCOPE_ANSWER = (
    "I'm designed to answer questions about Leadonspot "
    "and its products. I can't answer questions about "
    "other companies or unrelated topics."
)


# ==================================================
# 2. LOAD GROQ API KEY
# ==================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in backend .env file")

client = Groq(api_key=api_key)


# ==================================================
# 3. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


# ==================================================
# 4. CONNECT TO CHROMADB
# ==================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)

total_records = vectorstore._collection.count()

print("Total ChromaDB records:", total_records)

if total_records == 0:
    raise ValueError(
        "ChromaDB is empty. Run python ingest.py first."
    )


# ==================================================
# 5. PRICING QUESTION DETECTION
# ==================================================

PRICING_KEYWORDS = [
    "price",
    "pricing",
    "plan",
    "plans",
    "cost",
    "subscription",
    "subscriptions",
    "how much",
    "how many credits",
    "credit pack",
    "credit packs",
    "essential",
    "professional",
    "enterprise elite",
    "free trial",
    "trial period"
]


def is_pricing_question(question):

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in PRICING_KEYWORDS
    )


def is_future_pricing_question(question):

    question_lower = question.lower()

    future_year = re.search(
        r"\b20\d{2}\b",
        question_lower
    )

    future_words = [
        "future price",
        "future pricing",
        "next year",
        "in the future"
    ]

    if future_year:
        year = int(future_year.group())

        if year > 2026:
            return True

    return any(
        word in question_lower
        for word in future_words
    )


# ==================================================
# 6. PRIVACY AND CONFIDENTIALITY CHECK
# ==================================================

def handle_sensitive_question(question):

    q = question.lower()

    private_contact_patterns = [
        "personal mobile number",
        "personal phone number",
        "private phone number",
        "founder mobile number",
        "founder's mobile number",
        "founder phone number",
        "founder's phone number",
        "personal contact number"
    ]

    confidential_patterns = [
        "confidential customer information",
        "private customer information",
        "customer personal information",
        "customer database",
        "customer passwords",
        "confidential business information"
    ]

    unreleased_patterns = [
        "unreleased features",
        "unreleased feature",
        "secret features",
        "confidential roadmap",
        "internal roadmap",
        "unreleased product details"
    ]

    if any(pattern in q for pattern in private_contact_patterns):

        return (
            "I'm unable to share someone's private contact "
            "information. Please use Leadonspot's official "
            "contact channels."
        )

    if any(pattern in q for pattern in confidential_patterns):

        return (
            "I'm sorry, but I cannot share confidential "
            "customer information or private business data."
        )

    if any(pattern in q for pattern in unreleased_patterns):

        return (
            "I cannot share confidential or unreleased product "
            "information. Please follow Leadonspot's official "
            "announcements for updates."
        )

    return None


# ==================================================
# 7. OUT-OF-SCOPE QUESTION CHECK
# ==================================================

def is_out_of_scope_question(question):

    q = question.lower().strip()

    # Questions asking for another company's CEO
    # are outside the Leadonspot knowledge base.
    ceo_patterns = [
        r"\bwho\s+is\s+(?:the\s+)?(?:current\s+)?ceo\s+of\s+(.+)",
        r"\bwho\s+is\s+(.+?)'s\s+ceo\b",
        r"\bwho\s+is\s+the\s+chief\s+executive\s+of\s+(.+)"
    ]

    for pattern in ceo_patterns:

        match = re.search(pattern, q)

        if match:

            company = match.group(1).strip(
                " ?.!,"
            )

            if "leadonspot" not in company:
                return True

    # Other clearly unrelated general knowledge questions
    unrelated_patterns = [
        r"\bcapital of\b",
        r"\bweather in\b",
        r"\bpopulation of\b",
        r"\bpresident of\b",
        r"\bprime minister of\b"
    ]

    if any(
        re.search(pattern, q)
        for pattern in unrelated_patterns
    ):
        return True

    return False


# ==================================================
# 8. CLEAN AI RESPONSE
# ==================================================

def clean_response(text):

    if not text:
        return ""

    # Remove bold Markdown
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )

    # Remove italic Markdown
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)",
        r"\1",
        text
    )

    # Remove headings
    text = re.sub(
        r"(?m)^#{1,6}\s*",
        "",
        text
    )

    # Remove bullet markers
    text = re.sub(
        r"(?m)^\s*[-*]\s+",
        "",
        text
    )

    # Remove inline code
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text
    )

    # Remove extra blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==================================================
# 9. CREATE PROMPT
# ==================================================

def create_prompt(context, question):

    prompt = f"""
You are Leadonspot's AI assistant.

Your job is to answer customer questions about
Leadonspot and its products.

Use the following knowledge-base information
to answer the user's question.

KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:

ANSWERING FROM THE KNOWLEDGE BASE:

1. Use only the information provided in the context.
2. If the answer is available in the context,
   answer directly using that information.
3. Never say information is unavailable when
   it is clearly present in the context.
4. Never invent prices, features, integrations,
   discounts or product capabilities.
5. For pricing questions, clearly mention the
   relevant plan names and prices.
6. Include applicable GST information when provided.
7. For questions about plans, list plans on separate lines.
8. Do not combine features of different products.
9. Do not repeat the user's question.

UNKNOWN KNOWLEDGE AND HALLUCINATION:

10. If the requested information is genuinely missing
    from the context, politely say:
    "I don't have verified information about that.
    Please contact the Leadonspot team for confirmation."

11. Never guess or make assumptions to fill gaps
    in the knowledge base.
12. Do not use general knowledge to answer questions
    outside Leadonspot documentation.
13. Do not invent company office locations,
    partnerships, customer details or business information.
14. Never confirm an integration or partnership unless
    it is explicitly supported by the context.
15. Do not predict or invent future prices.
16. Leadonspot does not guarantee 100% lead conversion.
    If asked, clearly explain that it cannot guarantee
    100% conversion.
17. Never claim you checked external sources or contacted
    employees unless that actually happened.

PRIVACY AND CONFIDENTIALITY:

18. Never share personal mobile numbers, private
    contact details, passwords or sensitive information.
19. Never disclose confidential customer information
    or private business data.
20. Never reveal unreleased features, confidential
    product roadmaps or internal company information.
21. Politely refuse requests for private or confidential
    information.

RESPONSE STYLE:

22. Do not use Markdown formatting.
23. Do not use **, *, #, or bullet symbols.
24. Use plain text only.
25. For Yes/No questions, start with Yes or No
    whenever appropriate.
26. Use simple, friendly, everyday English.
27. Keep normal answers to a maximum of 2 sentences.
28. For pricing questions, provide enough detail
    to answer clearly.
29. Be polite, professional and direct.
30. Never make up an answer just to satisfy the user.

Provide the answer now:
"""

    return prompt


# ==================================================
# 10. GET CHATBOT RESPONSE
# ==================================================

def get_chatbot_response(question):

    question = question.strip()

    if not question:
        return "Please enter a question."

    print("\nQuestion received:", question)

    # ----------------------------------------------
    # A. PRIVATE AND CONFIDENTIAL QUESTIONS
    # ----------------------------------------------

    sensitive_answer = handle_sensitive_question(question)

    if sensitive_answer:
        print("Privacy rule applied.")
        return sensitive_answer

    # ----------------------------------------------
    # B. OUT-OF-SCOPE QUESTIONS
    # ----------------------------------------------

    if is_out_of_scope_question(question):

        print("Out-of-scope question detected.")

        return OUT_OF_SCOPE_ANSWER

    # ----------------------------------------------
    # C. FUTURE PRICING QUESTIONS
    # ----------------------------------------------

    if is_pricing_question(question) and is_future_pricing_question(question):

        print("Future pricing question detected.")

        return (
            "I don't have verified pricing for that year. "
            "Please contact the Leadonspot sales team."
        )

    # ----------------------------------------------
    # D. PRICING QUESTIONS
    # ----------------------------------------------

    if is_pricing_question(question):

        print("\nPricing question detected.")

        if not PRICING_FILE.exists():

            print("pricing.txt not found.")

            return (
                "I couldn't access the pricing information. "
                "Please contact our sales team."
            )

        with open(
            PRICING_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            context = file.read()

        if not context.strip():

            return (
                "I couldn't find verified pricing information. "
                "Please contact our sales team."
            )

        print("Pricing information loaded from:", PRICING_FILE)

    # ----------------------------------------------
    # E. OTHER QUESTIONS - CHROMADB RETRIEVAL
    # ----------------------------------------------

    else:

        print("\nSearching ChromaDB...")

        try:

            results = vectorstore.similarity_search_with_relevance_scores(
                question,
                k=5
            )

        except Exception as error:

            print("Retrieval error:", error)

            return (
                "I'm unable to retrieve verified information "
                "right now. Please try again."
            )

        relevant_documents = [
            (document, score)
            for document, score in results
            if score >= RELEVANCE_THRESHOLD
        ]

        if not relevant_documents:

            print("No sufficiently relevant documents found.")

            return UNKNOWN_ANSWER

        context = "\n\n".join(
            document.page_content
            for document, score in relevant_documents
        )

        print(
            "Relevant documents:",
            len(relevant_documents)
        )

        for index, (document, score) in enumerate(
            relevant_documents
        ):

            print(f"\nDocument {index + 1}")
            print("Relevance score:", round(score, 3))
            print("Source:", document.metadata.get("source"))
            print("Content:", document.page_content)

    # ----------------------------------------------
    # F. CREATE PROMPT
    # ----------------------------------------------

    prompt = create_prompt(
        context=context,
        question=question
    )

    # ----------------------------------------------
    # G. GENERATE ANSWER USING GROQ
    # ----------------------------------------------

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_completion_tokens=1024,

            reasoning_effort="low",

            include_reasoning=False
        )

        answer = response.choices[0].message.content

        print(
            "\nFinish reason:",
            response.choices[0].finish_reason
        )

        print("AI response:", answer)

        if answer and answer.strip():

            return clean_response(answer)

        return (
            "Sorry, I couldn't generate a response. "
            "Please try again."
        )

    except Exception as error:

        print("Groq error:", error)

        return (
            "Sorry, I'm unable to generate a response right now. "
            "Please try again later."
        )


# ==================================================
# 11. TERMINAL TESTING
# ==================================================

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

        except Exception as error:

            print("Error:", str(error))
