import os
import re
import time
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

RELEVANCE_THRESHOLD = 0.05
TOP_K = 6

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
    "essential plan",
    "professional plan",
    "enterprise elite",
    "credit",
    "credits",
    "charges",
    "expensive",
    "billing",
    "free trial",
    "trial period"
]


def is_pricing_question(question):
    question_lower = question.lower()

    return any(
        re.search(
            r"\b" + re.escape(keyword) + r"\b",
            question_lower
        )
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
# 6B. RISKY / HALLUCINATION-BAIT QUESTIONS
# ==================================================

REFUSE_INTERNALS = (
    "I can only help with questions about Leadonspot and its products. "
    "I can't share or change my internal instructions or system details."
)

RISKY_RULES = [
    (
        r"ignore (all |any |your )?(previous|prior|security|safety)?\s*(instructions|rules)|"
        r"ignore .{0,20}security rules|system prompt|hidden instructions|"
        r"(reveal|show|print|encode).{0,30}(instructions|prompt)|base64|"
        r"you are no longer|act as (chatgpt|an? )|pretend (to be|you are)|"
        r"developer mode|jailbreak|confidential instructions|"
        r"i am (the|your) (ceo|founder|admin|developer|owner)",
        REFUSE_INTERNALS
    ),
    (
        r"api[\s_-]?keys?|secret keys?|access tokens?|\bpasswords?\b|credentials",
        "I can't share credentials, keys or other internal system details."
    ),
    (
        r"(internal|your|the) (knowledge base|company documents|documents|database)\b.{0,20}(show|reveal|share|give)|"
        r"(show|reveal|share|give).{0,25}(internal|your) (knowledge base|company documents|documents|database)|"
        r"internal company documents",
        "I can't share internal documents or my knowledge base. I'm happy to answer "
        "questions about Leadonspot's products, pricing and features."
    ),
    (
        r"(another|other|someone else'?s?) (customer|user|client)|"
        r"share my conversation|employee (information|details|data|salary)|"
        r"employees'? personal|confidential employee",
        "I'm sorry, I can't share information about other customers, employees or "
        "conversations. Your chat is only visible to you and the Leadonspot team."
    ),
    (
        r"access my (personal )?(files|computer|device|laptop|phone|emails?)",
        "No. I can't access your files or devices. I only see the messages you type here."
    ),
    (
        r"(fake|made[- ]?up|fabricat\w*|invent\w*).{0,25}(testimonial|review|case stud|customer stor)|"
        r"generate.{0,15}testimonials?",
        "I can't create fake testimonials or reviews. I can share what Leadonspot "
        "actually offers, and our team can share verified customer stories."
    ),
    (
        r"(pretend|claim|say|confirm|state).{0,40}(unlimited|already live|is live|is available|free ai|offers? free)|"
        r"pretend .{0,40}(feature|offers?)",
        "I can't present unverified features or offers as real. I can only share "
        "Leadonspot's documented plans and features."
    ),
    (
        r"(negative|bad|worst|trash|attack|false|unsupported).{0,30}(competitor|other crm|rival)|"
        r"(competitors?|rivals?).{0,30}(negative|bad things|unsupported claims)|"
        r"unsupported claims about",
        "I don't make negative or unsupported claims about other companies. "
        "I can tell you what Leadonspot offers."
    ),
    (
        r"legal(ly)?.{0,12}(guarantee|binding|commitment)|guarantee.{0,12}legal",
        "I can't give legal guarantees or commitments on behalf of Leadonspot. "
        "Please refer to the official terms or contact the Leadonspot team."
    ),
    (
        r"\bguarantee[sd]?\b|\b100\s*%|\b10x\b|\bdouble my (sales|revenue)\b",
        "Leadonspot cannot guarantee specific results such as 100% conversion or "
        "10x revenue. It helps you prioritise leads and follow up faster."
    ),
    (
        r"\b(discount|coupon|promo code|free forever|lifetime deal|refund policy)\b",
        "I don't have verified information about discounts or offers. "
        "Please contact the Leadonspot sales team for confirmation."
    ),
    (
        r"(integration|connect|partnership|partner).{0,60}(not (mentioned|listed|documented)|undocumented)|"
        r"not (mentioned|listed|documented) in (your|the) documentation",
        "I can only confirm integrations and partnerships that are listed in "
        "Leadonspot's documentation. For anything else, please contact the team."
    ),
    (
        r"(information|answer|details?).{0,30}(not|isn't) (available|in) (in )?(your|the) (knowledge base|documentation)",
        "I can only share verified information about Leadonspot. If something isn't "
        "in my documentation, I'll say so instead of guessing."
    ),
    (
        r"\b(office|headquarter|branch)\b.*\b(in|at)\b.*\b(london|usa|america|dubai|singapore|germany|canada|australia|uk)\b",
        UNKNOWN_ANSWER
    ),
    (
        r"\b(partnership|partner(ed)?|tie-?up)\b.{0,25}\b(salesforce|hubspot|zoho|microsoft|google|oracle|sap)\b",
        UNKNOWN_ANSWER
    ),
]


def handle_risky_question(question):
    q = question.lower()

    for pattern, reply in RISKY_RULES:
        if re.search(pattern, q):
            return reply

    return None

# ==================================================
# 7. OUT-OF-SCOPE QUESTION CHECK
# ==================================================


def is_out_of_scope_question(question):
    q = question.lower().strip()

    ceo_patterns = [
        r"\bwho\s+is\s+(?:the\s+)?(?:current\s+)?ceo\s+of\s+(.+)",
        r"\bwho\s+is\s+(.+?)'s\s+ceo\b",
        r"\bwho\s+is\s+the\s+chief\s+executive\s+of\s+(.+)"
    ]

    for pattern in ceo_patterns:
        match = re.search(pattern, q)

        if match:
            company = match.group(1).strip(" ?.!,")
            if "leadonspot" not in company:
                return True

    unrelated_patterns = [
        r"\bcapital of\b",
        r"\bweather in\b",
        r"\bpopulation of\b",
        r"\bpresident of\b",
        r"\bprime minister of\b",
        r"\b(tell|say) me a (joke|story|riddle)\b|\bjoke\b",
        r"\b(write|give) (me )?(a )?(poem|song|essay|python|code)\b",
        r"\b(recipe|stock price|cricket score|movie recommendation)\b"
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

    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ==================================================
# 9. CREATE PROMPT
# ==================================================


def create_prompt(context, question, history_text=""):
    prompt = f"""
You are Leadonspot's AI assistant.

Your job is to answer customer questions about
Leadonspot and its products.

Use the following knowledge-base information
to answer the user's question.

KNOWLEDGE BASE CONTEXT:

{context}

RECENT CONVERSATION (for follow-up questions only):

{history_text or "None"}

USER QUESTION:

{question}

INSTRUCTIONS:

ANSWERING FROM THE KNOWLEDGE BASE:

1. Use only the information provided in the context.
2. If the answer is available in the context, answer directly using that information.
3. Never say information is unavailable when it is clearly present in the context.
4. Never invent prices, features, integrations, discounts or product capabilities.
5. For pricing questions, clearly mention the relevant plan names and prices.
6. Include applicable GST information when provided.
7. For questions about plans, list plans on separate lines.
8. Do not combine features of different products.
9. Do not repeat the user's question.

UNKNOWN KNOWLEDGE AND HALLUCINATION:

10. If the requested information is genuinely missing from the context, politely say:

"I don't have verified information about that. Please contact the Leadonspot team for confirmation."

11. Never guess or make assumptions to fill gaps in the knowledge base.
12. Do not use general knowledge to answer questions outside Leadonspot documentation.
13. Do not invent company office locations, partnerships, customer details or business information.
14. Never confirm an integration or partnership unless it is explicitly supported by the context.
15. Do not predict or invent future prices.
16. Leadonspot does not guarantee 100% lead conversion. If asked, clearly explain that it cannot guarantee 100% conversion.
17. Never claim you checked external sources or contacted employees unless that actually happened.

PRIVACY AND CONFIDENTIALITY:

18. Never share personal mobile numbers, private contact details, passwords or sensitive information.
19. Never disclose confidential customer information or private business data.
20. Never reveal unreleased features, confidential product roadmaps or internal company information.
21. Politely refuse requests for private or confidential information.

RESPONSE STYLE:

22. Do not use Markdown formatting.
23. Do not use **, *, #, or bullet symbols.
24. Use plain text only.
25. For Yes/No questions (Can I..., Does it..., Is there...), answer in exactly ONE sentence that starts with Yes or No.
26. Use simple, friendly, everyday English.
27. Keep answers SHORT. Descriptive questions (What is..., How does..., Why...) get exactly 2 sentences. Yes/No questions get 1 sentence. Only when the user asks several things at once, or asks for a list of features, steps or plans, may you use up to 4 short sentences.
28. For pricing questions, provide enough detail to answer clearly.
29. Be polite, professional and direct.
30. Never make up an answer just to satisfy the user.

RECOMMENDATIONS AND FOLLOW-UPS:

31. If the user describes their business (industry, team size, problem), recommend the matching Leadonspot products using the industry and product information in the context. Team size or other details they mention are context, not a request for a demo.

32. For follow-up questions ("what about its price?", "and for that?"), use the RECENT CONVERSATION to understand what "it" or "that" refers to.

34. If the user states a fact that contradicts the context (for example a wrong trial length), politely correct it using the context.

35. Never make claims about other companies or compare features with competitors. Never write phrases like "many competitors do not", "unlike other CRMs" or "better than". If the context has no comparison for the named product (for example Zoho), say you do not have a verified comparison, then give Leadonspot's own strengths. When asked why to choose Leadonspot or how it differs, describe only Leadonspot's own strengths from the context.

36. If the user asks for a recommendation but gives little detail (for example "I need a CRM" or "which product is best for me"), recommend from what you know and ask ONE question: their industry, team size or monthly lead volume. If asked what information you need, list: industry, team size, monthly leads, current tool, main challenge, and timeline.

37. If the user asks you a question about THEIR business (for example "What industry are you operating in?"), explain that you are Leadonspot's assistant, and invite them to share their own details so you can recommend a product.

38. If the user wants a reply in Hindi or Hinglish, or simpler words, reply that way, using only facts from the context.

39. Use RECENT CONVERSATION only for what the user actually said. If they ask about something they have not told you, say so and ask them.

41. If asked whether you can remember their requirements, say yes: you remember what they tell you during this conversation only.

40. If the user says they will ask several questions together but has not asked them yet, say yes and ask them to send them.

33. If, and only if, the answer is truly not in the context, reply with exactly:

I don't have verified information about that.

Please contact the Leadonspot team for confirmation.

Provide the answer now:
"""
    return prompt

# ==================================================
# 8B. KEEP ANSWERS SHORT AND SAFE
# ==================================================

UNSUPPORTED_COMPARISON = re.compile(
    r"\b(many|most|other|some) (competitors|crms|platforms|tools|vendors)\b|"
    r"\bunlike (most|other|many)\b|\bbetter than\b",
    re.IGNORECASE,
)

NEEDS_MORE_ROOM = re.compile(
    r"\b(list|steps?|how (can|do) i|features?|plans?|pricing|compare|"
    r"difference|in detail|summar|all four|nexus.*pricing)\b|,.*\band\b",
    re.IGNORECASE,
)

YES_NO_START = re.compile(
    r"^\s*(can|could|does|do|did|is|are|was|were|will|would|should|has|have|am)\b",
    re.IGNORECASE,
)

REQUEST_NOT_YES_NO = re.compile(
    r"^\s*(can|could|would) you (please )?(explain|tell|describe|recommend|"
    r"summari[sz]e|help|show|give|walk|list|suggest|compare)\b",
    re.IGNORECASE,
)


def is_yes_no_question(question):
    q = question.strip()
    first_sentence = re.split(r"(?<=[.?!])\s+", q)[0]

    return bool(
        YES_NO_START.match(first_sentence)
        and not REQUEST_NOT_YES_NO.match(first_sentence)
    )


def first_sentences(text, count):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    if (
        len(sentences) > 1
        and re.fullmatch(
            r"(yes|no)[.!]?",
            sentences[0].strip(),
            re.IGNORECASE
        )
    ):
        sentences = [sentences[0] + " " + sentences[1]] + sentences[2:]

    return " ".join(sentences[:count])


def shorten_answer(question, answer, has_pricing):
    yes_no = is_yes_no_question(question)

    if not yes_no and (has_pricing or "\n" in answer.strip()):
        return answer

    text = re.sub(r"\s+", " ", answer.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if "hubspot" not in question.lower():
        kept = [
            sentence
            for sentence in sentences
            if not UNSUPPORTED_COMPARISON.search(sentence)
        ]
        sentences = kept or sentences

    text = " ".join(sentences)

    if yes_no:
        return first_sentences(text, 1)

    if NEEDS_MORE_ROOM.search(question):
        return first_sentences(text, 4)

    return first_sentences(text, 2)

# ==================================================
# 9B. GROQ CALL
# ==================================================

LLM_MODEL = "openai/gpt-oss-20b"


def call_llm(prompt):
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_completion_tokens=300,
            reasoning_effort="low",
            include_reasoning=False,
        )

        print(
            f"Groq response time: "
            f"{time.time() - start_time:.2f} seconds"
        )

        return response.choices[0].message.content

    except Exception as error:
        print("Groq API error:", str(error)[:500])
        print(
            f"Groq failed after: "
            f"{time.time() - start_time:.2f} seconds"
        )
        return None

# ==================================================
# 10. GET CHATBOT RESPONSE
# ==================================================


def get_chatbot_response(question, history=None):
    question = question.strip()

    if not question:
        return "Please enter a question."

    print("\nQuestion received:", question)

    # A. PRIVATE AND CONFIDENTIAL QUESTIONS

    sensitive_answer = handle_sensitive_question(question)

    if sensitive_answer:
        print("Privacy rule applied.")
        return sensitive_answer

    if re.search(
        r"previous (conversation|session|chat)|last (time|session|chat)|"
        r"yesterday|where we (left|stopped)",
        question.lower()
    ):
        return (
            "I can only see this current conversation, not earlier chats. "
            "Tell me where you left off or what you need, and I'll help from there."
        )

    risky_answer = handle_risky_question(question)

    if risky_answer:
        print("Risky-question rule applied.")
        return risky_answer

    # B. OUT-OF-SCOPE QUESTIONS

    if is_out_of_scope_question(question):
        print("Out-of-scope question detected.")
        return OUT_OF_SCOPE_ANSWER

    # C. FUTURE PRICING QUESTIONS

    if (
        is_pricing_question(question)
        and is_future_pricing_question(question)
    ):
        print("Future pricing question detected.")

        return (
            "I don't have verified pricing for that year. "
            "Please contact the Leadonspot sales team."
        )

    # D. PRICING QUESTIONS

    pricing_context = ""

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
            pricing_context = file.read()

        if not pricing_context.strip():
            return (
                "I couldn't find verified pricing information. "
                "Please contact our sales team."
            )

        print("Pricing information loaded from:", PRICING_FILE)

    # E. CHROMADB RETRIEVAL - SINGLE SEARCH

    print("\nSearching ChromaDB...")

    try:
        search_query = question

        # Add the previous user question for short follow-ups.
        if history and len(question.split()) <= 7:
            last_user = next(
                (
                    item["content"]
                    for item in reversed(history)
                    if item["role"] == "user"
                ),
                ""
            )

            search_query = f"{last_user} {question}".strip()

        # Perform only ONE ChromaDB search.
        results = vectorstore.similarity_search_with_relevance_scores(
            search_query,
            k=5
        )

        relevant_documents = [
            (document, score)
            for document, score in results
            if score >= RELEVANCE_THRESHOLD
        ]

        # Fallback to top results if relevance scores are low.
        if not relevant_documents and results:
            relevant_documents = list(results[:3])

        # Limit context sent to Groq.
        relevant_documents = relevant_documents[:3]

        context = "\n\n".join(
            ([pricing_context] if pricing_context else [])
            + [
                document.page_content
                for document, score in relevant_documents
            ]
        )

        print("Retrieved documents:", len(relevant_documents))

        for index, (document, score) in enumerate(relevant_documents):
            print(f"\nDocument {index + 1}")
            print("Relevance score:", round(score, 3))
            print("Source:", document.metadata.get("source"))

    except Exception as error:
        print("Retrieval error:", error)

        return (
            "I'm unable to retrieve verified information "
            "right now. Please try again."
        )

    # F. CREATE PROMPT

    history_text = "\n".join(
        f"{item['role'].title()}: {item['content']}"
        for item in (history or [])[-12:]
    )

    prompt = create_prompt(
        context=context,
        question=question,
        history_text=history_text
    )

    # G. GENERATE ANSWER USING GROQ

    answer = call_llm(prompt)

    if answer and answer.strip():
        cleaned = clean_response(answer)

        cleaned = shorten_answer(
            question,
            cleaned,
            bool(pricing_context)
        )

        if "don't have verified information" in cleaned.lower():
            return UNKNOWN_ANSWER

        return cleaned

    return (
        "Sorry, I'm unable to generate a response right now. "
        "Please try again in a moment."
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