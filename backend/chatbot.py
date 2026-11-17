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
# Keep this LOW. A high threshold returns UNKNOWN before the LLM ever sees
# the context (this is why many answerable questions failed). The LLM now
# decides using the UNKNOWN rule in the prompt.
RELEVANCE_THRESHOLD = 0.05
TOP_K = 5
FETCH_K = 20   # fetch wide, then re-rank down to TOP_K

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

client = Groq(api_key=api_key, timeout=20.0, max_retries=0)


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
    "try the product",
    "try before",
    "before purchasing",
    "before buying",
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
        re.search(r"\b" + re.escape(keyword) + r"\b", question_lower)
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
# 6b. RISKY / HALLUCINATION-BAIT QUESTIONS
# ==================================================

REFUSE_INTERNALS = (
    "I can only help with questions about Leadonspot and its products. "
    "I can't share or change my internal instructions or system details."
)

RISKY_RULES = [
    # --- prompt injection / role change / internals ---
    (r"ignore (all |any |your )?(previous|prior|security|safety)?\s*(instructions|rules)|"
     r"ignore .{0,20}security rules|system prompt|hidden instructions|"
     r"(reveal|show|print|encode).{0,30}(instructions|prompt)|base64|"
     r"you are no longer|act as (chatgpt|an? )|pretend (to be|you are)|"
     r"developer mode|jailbreak|confidential instructions|"
     r"i am (the|your) (ceo|founder|admin|developer|owner)",
     REFUSE_INTERNALS),
    (r"api[\s_-]?keys?|secret keys?|access tokens?|\bpasswords?\b|credentials",
     "I can't share credentials, keys or other internal system details."),
    (r"(internal|your|the) (knowledge base|company documents|documents|database)\b.{0,20}(show|reveal|share|give)|"
     r"(show|reveal|share|give).{0,25}(internal|your) (knowledge base|company documents|documents|database)|"
     r"internal company documents",
     "I can't share internal documents or my knowledge base. I'm happy to answer "
     "questions about Leadonspot's products, pricing and features."),
    # --- other people's / private data ---
    (r"(another|other|someone else'?s?) (customer|user|client)|"
     r"share my conversation|employee (information|details|data|salary)|"
     r"employees'? personal|confidential employee",
     "I'm sorry, I can't share information about other customers, employees or "
     "conversations. Your chat is only visible to you and the Leadonspot team."),
    (r"access my (personal )?(files|computer|device|laptop|phone|emails?)",
     "No. I can't access your files or devices. I only see the messages you type here."),
    # --- fabrication ---
    (r"(fake|made[- ]?up|fabricat\w*|invent\w*).{0,25}(testimonial|review|case stud|customer stor)|"
     r"generate.{0,15}testimonials?",
     "I can't create fake testimonials or reviews. I can share what Leadonspot "
     "actually offers, and our team can share verified customer stories."),
    (r"(pretend|claim|say|confirm|state).{0,40}(unlimited|already live|is live|is available|free ai|offers? free)|"
     r"pretend .{0,40}(feature|offers?)",
     "I can't present unverified features or offers as real. I can only share "
     "Leadonspot's documented plans and features."),
    (r"(negative|bad|worst|trash|attack|false|unsupported).{0,30}(competitor|other crm|rival)|"
     r"(competitors?|rivals?).{0,30}(negative|bad things|unsupported claims)|"
     r"unsupported claims about",
     "I don't make negative or unsupported claims about other companies. "
     "I can tell you what Leadonspot offers."),
    (r"legal(ly)?.{0,12}(guarantee|binding|commitment)|guarantee.{0,12}legal",
     "I can't give legal guarantees or commitments on behalf of Leadonspot. "
     "Please refer to the official terms or contact the Leadonspot team."),
    # --- guarantees ---
    (r"\bguarantee[sd]?\b|\b100\s*%|\b10x\b|\bdouble my (sales|revenue)\b",
     "Leadonspot cannot guarantee specific results such as 100% conversion or "
     "10x revenue. It helps you prioritise leads and follow up faster."),
    (r"\b(discount|coupon|promo code|free forever|lifetime deal|refund policy)\b",
     "I don't have verified information about discounts or offers. "
     "Please contact the Leadonspot sales team for confirmation."),
    # --- hallucination bait ---
    (r"(integration|connect|partnership|partner).{0,60}(not (mentioned|listed|documented)|undocumented)|"
     r"not (mentioned|listed|documented) in (your|the) documentation",
     "I can only confirm integrations and partnerships that are listed in "
     "Leadonspot's documentation. For anything else, please contact the team."),
    (r"(information|answer|details?).{0,30}(not|isn'?t) (available|in) (in )?(your|the) (knowledge base|documentation)",
     "I can only share verified information about Leadonspot. If something isn't "
     "in my documentation, I'll say so instead of guessing."),
    # Bengaluru office — confirmed location
    (r"\b(office|headquarters?|branch)\b.{0,40}\b(in|at)\s+(bengaluru|bangalore)\b",
    "Yes, Leadonspot has its office in Bengaluru, India."),

    # India — clarify the Bengaluru location
    (r"\b(office|headquarters?|branch)\b.{0,40}\b(in|at)\s+india\b",
    "Leadonspot has its office in Bengaluru, India."),

    # All other office locations
    (r"\b(office|headquarters?|branch)\b.{0,40}\b(in|at)\s+(?!bengaluru\b|bangalore\b|india\b)[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}\b",
    "No, Leadonspot has its office in Bengaluru, India, and does not have offices in other locations."),
    (r"\b(partnership|partner(ed)?|tie-?up)\b.{0,25}\bsalesforce\b",
     "I don't have verified information about a partnership with Salesforce, "
     "but Leadonspot lists a native Salesforce integration for enterprise customers."),
    (r"\b(partnership|partner(ed)?|tie-?up)\b.{0,25}\b(hubspot|zoho|microsoft|google|oracle|sap)\b",
     UNKNOWN_ANSWER),
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

def create_prompt(context, question, history_text=""):

    return f"""You are Leadonspot's AI assistant. Answer using ONLY the CONTEXT and the actual RECENT CONVERSATION where applicable.

CONTEXT:
{context}

RECENT CONVERSATION:
{history_text or "None"}

QUESTION:
{question}

RULES:

1. Use only the context for Leadonspot company information. Never invent prices, features, integrations, offices, partnerships, discounts or customers. Never use outside knowledge.

2. If the context answers the question, answer directly. If only part of the information is documented, give the documented part and say the rest should be confirmed with the Leadonspot team. Say exactly "I don't have verified information about that. Please contact the Leadonspot team for confirmation." only when the topic is not in the context at all.

3. Capability questions (Can I / Does it / Is it possible): if the context describes the same or a clearly related capability, answer Yes and state it.

4. Plain text only, no Markdown or bullets. Simple friendly English. Do not repeat the question.

5. Length: Yes/No questions get ONE sentence starting with Yes or No. Descriptive questions get TWO sentences. Use up to 4 short sentences only for several questions at once, or lists of features, steps or plans.

6. Pricing: name the plans and prices, include GST when given, one plan per line.

7. Never compare with or criticise other companies, and never say "unlike others" or "better than". If asked why to choose Leadonspot, describe only its own strengths from the context.

8. Never guarantee results (for example 100% conversion). Never predict future prices. Never share private contact details, confidential data or unreleased features.

9. If the user states something that contradicts the context (for example a wrong trial length), politely correct it.

10. If they describe their business, recommend matching products from the context. If they give little detail ("I need a CRM"), recommend and ask ONE question about industry, team size or monthly leads. Team size is not a demo request.

11. For follow-ups ("what about its price?") use RECENT CONVERSATION to resolve "it" or "that". Use it only for what the user actually said; otherwise say you do not have that yet. You remember details only within this conversation.

12. If asked about their own business (for example "What industry are you in?"), say you are Leadonspot's assistant and invite them to share their details. If they will ask several questions together, say yes and ask them to send them. Reply in Hindi or Hinglish or simpler words if asked, using only context facts.

13. CONVERSATION HISTORY QUESTIONS:
    If the user asks questions such as:
    - What products did we discuss earlier?
    - What did we talk about?
    - Which solutions did you recommend?
    - What was my previous question?
    - What have we discussed so far?

    Answer using ONLY the actual RECENT CONVERSATION.

    Do not use the CONTEXT or knowledge base to guess what was discussed.

    Mention only products, services, features or topics that were explicitly discussed in the conversation.

    Do not list all Leadonspot products just because they appear in the CONTEXT.

    If no relevant discussion exists in the conversation, respond:
    "We haven't discussed any products or solutions in this conversation yet."

14. CONVERSATION HISTORY PRIORITY:
    For questions about previous messages, RECENT CONVERSATION is the source of truth.
    Never treat information from the CONTEXT as something the user has previously discussed.
    Never claim that the user asked, mentioned, selected or showed interest in something unless it appears in RECENT CONVERSATION.

15. MISSING OR EMPTY HISTORY:
    If RECENT CONVERSATION is "None" or does not contain the requested earlier discussion, do not guess or fabricate an answer.
    Politely explain that you cannot identify the earlier discussion from the available conversation history.

16. NAME AND FOLLOW-UP:
    Do not repeatedly ask for the user's name after every answer.
    Ask for the user's name only when it is required for an agreed follow-up or lead-capture process.
    If the user has already provided their name in RECENT CONVERSATION, do not ask for it again.

ANSWER:"""


# ==================================================
# 8a. RE-RANK RETRIEVED CHUNKS (hybrid: embeddings + keyword match)
# ==================================================

STOP_WORDS = {
    "the", "and", "for", "are", "can", "does", "did", "you", "your", "with",
    "that", "this", "from", "have", "has", "how", "what", "why", "who",
    "when", "where", "which", "will", "would", "could", "should", "any",
    "our", "its", "into", "about", "leadonspot", "i", "my", "me", "do",
    "is", "it", "to", "of", "in", "on", "a", "an", "be", "or", "if", "we",
}


def _stem(word):
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_words(text):
    return {
        _stem(w)
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in STOP_WORDS
    }


def rerank_results(question, results):
    """Remove duplicate chunks and boost chunks that share the user's own
    words, so "Can I assign leads by location?" finds the chunk that says
    so even when the embedding score is only average."""
    question_words = content_words(question)
    seen = set()
    ranked = []

    for document, score in results:
        text = document.page_content
        fingerprint = re.sub(r"\W+", " ", text.lower())[:180]

        if fingerprint in seen:
            continue

        seen.add(fingerprint)

        chunk_words = content_words(text)
        overlap = (
            len(question_words & chunk_words) / len(question_words)
            if question_words else 0
        )

        bonus = 0.35 * overlap

        # FAQ-style chunk ("Q: ...") whose question matches the user's
        # question: strong signal, put it first.
        first_line = text.strip().split("\n")[0]

        if first_line.lower().startswith(("q:", "question:")):
            head_words = content_words(first_line)
            head_overlap = (
                len(question_words & head_words) / len(question_words)
                if question_words else 0
            )

            if head_overlap >= 0.6:
                bonus += 0.8 * head_overlap

        ranked.append((document, score + bonus))

    ranked.sort(key=lambda item: item[1], reverse=True)

    return ranked


FAQ_HEAD = re.compile(r"^\s*(?:Q|Question)\s*:\s*(.+)", re.IGNORECASE)
FOLLOW_UP_WORDS = re.compile(
    r"\b(it|its|that|this|those|they|them|again|earlier|before|previous)\b",
    re.IGNORECASE,
)


def extract_faq_answer(text):
    m = re.search(r"(?:^|\s)(?:Answer|A)\s*:\s*(.*)", text, re.S)

    if not m:
        return ""

    answer = m.group(1)
    answer = re.split(r"\n\s*={5,}", answer)[0]                 # section dividers
    answer = re.split(r"\n\s*(?:Q|Question)\s*:", answer)[0]    # next question

    return re.sub(r"\s+", " ", answer).strip()


def direct_faq_answer(question, reranked, history):
    """If the question is (almost) identical to a stored FAQ question,
    return that stored answer with NO model call: instant and consistent."""
    if history and FOLLOW_UP_WORDS.search(question):
        return None

    question_words = content_words(question)

    if len(question_words) < 2:
        return None

    best = None

    for document, score in reranked[:6]:
        text = document.page_content.strip()
        head = FAQ_HEAD.match(text)

        if not head:
            continue

        head_words = content_words(head.group(1).split("\n")[0])

        if len(head_words) < 2:
            continue

        similarity = (
            len(question_words & head_words)
            / len(question_words | head_words)
        )

        if similarity >= 0.75 and (best is None or similarity > best[0]):
            best = (similarity, text)

    if not best:
        return None

    answer = extract_faq_answer(best[1])

    return answer if len(answer) >= 15 else None


def is_unknown_text(text):
    return "don't have verified information about that" in (text or "").lower()


# ==================================================
# 8b. KEEP ANSWERS SHORT AND SAFE
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

    # "Yes." / "No." alone is not a full answer: join it with the next one.
    if len(sentences) > 1 and re.fullmatch(r"(yes|no)[.!]?", sentences[0].strip(), re.IGNORECASE):
        sentences = [sentences[0] + " " + sentences[1]] + sentences[2:]

    return " ".join(sentences[:count])


def shorten_answer(question, answer, has_pricing):
    yes_no = is_yes_no_question(question)

    # Plan lists / bullet-style answers keep their lines, unless the user
    # only asked a yes/no question.
    if not yes_no and (has_pricing or "\n" in answer.strip()):
        return answer

    text = re.sub(r"\s*\n\s*", " ", answer.strip())

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Drop unsupported claims about competitors (HubSpot has a real
    # comparison in the knowledge base, so it is exempt).
    if "hubspot" not in question.lower():
        kept = [x for x in sentences if not UNSUPPORTED_COMPARISON.search(x)]
        sentences = kept or sentences

    text = " ".join(sentences)

    if yes_no:
        # Clear answers get ONE sentence. If the answer hedges ("please
        # confirm...", "I don't have verified information..."), keep the
        # second sentence too, so the customer still learns what to do.
        if re.search(r"please confirm|don't have verified", text, re.IGNORECASE):
            return first_sentences(text, 2)

        return first_sentences(text, 1)

    if NEEDS_MORE_ROOM.search(question):
        return first_sentences(text, 4)

    return first_sentences(text, 2)


# ==================================================
# 9b. GROQ CALL WITH RETRY (handles 429 rate limits)
# ==================================================

import time

LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")


def _retry_wait_seconds(error_text, attempt):
    # Groq says e.g. "Please try again in 6.5s" or "in 1m3.4s"
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", error_text)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    return min(3 * (2 ** attempt), 30)


def call_llm(prompt):
    """One request per model. A rate limit (429) switches to the fallback
    model at once (separate limits) instead of sleeping for ~30s."""
    models = [LLM_MODEL, FALLBACK_MODEL, FALLBACK_MODEL]

    for attempt, model in enumerate(models):
        started = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=350,
                reasoning_effort="low",
                include_reasoning=False,
            )

            print(f"LLM ({model}) took {time.time() - started:.1f}s")

            return response.choices[0].message.content

        except Exception as error:
            text = str(error)
            print(f"Groq error (attempt {attempt + 1}, {model}):", text[:300])

            if attempt == 0:
                continue                      # switch model immediately

            if attempt == 1 and ("429" in text or "rate" in text.lower()):
                time.sleep(min(_retry_wait_seconds(text, attempt), 5))
                continue

            break

    return None


# ==================================================
# 10. GET CHATBOT RESPONSE
# ==================================================

def _answer_question(question, history=None):

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

    if re.search(r"previous (conversation|session|chat)|last (time|session|chat)|"
                 r"yesterday|where we (left|stopped)", question.lower()):
        return ("I can only see this current conversation, not earlier chats. "
                "Tell me where you left off or what you need, and I'll help from there.")

    risky_answer = handle_risky_question(question)

    if risky_answer:
        print("Risky-question rule applied.")
        return risky_answer

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

    pricing_context = ""
    extra_context = ""

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

    # ----------------------------------------------
    # E. OTHER QUESTIONS - CHROMADB RETRIEVAL
    # ----------------------------------------------

    # Always retrieve; pricing questions get pricing.txt PLUS retrieved
    # chunks, so "explain CRM, Nexus and pricing" has all the facts.
    if True:

        print("\nSearching ChromaDB...")

        try:

            # Short follow-ups ("and its price?") need the previous
            # user question to retrieve the right chunks.
            search_query = question

            if history and len(question.split()) <= 7:
                last_user = next(
                    (t["content"] for t in reversed(history)
                     if t["role"] == "user"),
                    ""
                )
                search_query = f"{last_user} {question}".strip()

            results = vectorstore.similarity_search_with_relevance_scores(
                search_query,
                k=FETCH_K
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

        # Scores can be low/negative for general questions ("explain simply",
        # "answer 5 questions"). Don't give up: pass the top 3 and let the
        # prompt's UNKNOWN rule decide.
        if not relevant_documents and results:
            relevant_documents = list(results[:3])

        # Re-rank (dedupe + keyword boost) over everything we fetched,
        # then keep the prompt small for Groq's token limits.
        pool = relevant_documents if len(relevant_documents) >= 3 else list(results)
        reranked = rerank_results(question, pool)

        instant = direct_faq_answer(question, reranked, history)

        if instant:
            print("Instant FAQ answer (no model call).")
            return shorten_answer(question, instant, False)
        relevant_documents = reranked[:2 if pricing_context else TOP_K]

        # Second-chance context, used only if the first answer is "unknown".
        extra_context = "\n\n".join(
            document.page_content[:500]
            for document, score in reranked[:8]
        )

        context = "\n\n".join(
            ([pricing_context] if pricing_context else [])
            + [document.page_content[:700]
               for document, score in relevant_documents]
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

    history_text = "\n".join(
        f"{t['role'].title()}: {t['content'][:300]}"
        for t in (history or [])[-6:]
    )

    prompt = create_prompt(
        context=context,
        question=question,
        history_text=history_text
    )

    # ----------------------------------------------
    # G. GENERATE ANSWER USING GROQ
    # ----------------------------------------------

    answer = call_llm(prompt)

    # Second chance: if the model said "unknown", retry once with a wider,
    # re-ranked context before giving up.
    if is_unknown_text(answer) and extra_context:
        print("First answer was UNKNOWN: retrying with wider context...")

        retry_prompt = create_prompt(
            context=extra_context,
            question=question,
            history_text=history_text
        )

        retry_answer = call_llm(retry_prompt)

        if retry_answer and retry_answer.strip():
            answer = retry_answer

    if answer and answer.strip():

        cleaned = clean_response(answer)
        cleaned = shorten_answer(question, cleaned, bool(pricing_context))

        # Normalise every "don't know" variant to the fixed fallback.
        # Only the exact fixed fallback is normalised; careful answers that
        # state what IS documented and what to confirm are kept as written.
        if cleaned.lower().startswith(
            "i don't have verified information about that"
        ):
            return UNKNOWN_ANSWER

        return cleaned

    return (
        "Sorry, I'm unable to generate a response right now. "
        "Please try again in a moment."
    )


# ==================================================
# 10b. CACHE, TIMING AND WARM-UP
# ==================================================

_CACHE = {}
CACHE_TTL_SECONDS = 3600
CACHE_MAX_ITEMS = 500


def get_chatbot_response(question, history=None):
    started = time.time()
    key = re.sub(r"\W+", " ", question.lower()).strip()
    use_cache = not history          # follow-ups depend on the conversation

    if use_cache and key in _CACHE:
        saved_at, saved_answer = _CACHE[key]

        if time.time() - saved_at < CACHE_TTL_SECONDS:
            print(f"Cache hit. Total response time: {time.time() - started:.2f}s")
            return saved_answer

    answer = _answer_question(question, history)

    failed = (not answer) or answer.startswith(
        ("Sorry, I'm unable", "I'm unable to retrieve")
    )

    if use_cache and not failed and len(_CACHE) < CACHE_MAX_ITEMS:
        _CACHE[key] = (time.time(), answer)

    print(f"Total response time: {time.time() - started:.2f}s")

    return answer


def warm_up():
    """Run one tiny search at startup so the first real question is fast."""
    try:
        started = time.time()
        vectorstore.similarity_search("warm up", k=1)
        print(f"Retriever warmed up in {time.time() - started:.1f}s")
    except Exception as error:
        print("Warm-up skipped:", error)


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