
import sqlite3
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "leads.db"


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

def initialize_database():

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                intent TEXT,
                details TEXT,
                current_step TEXT,
                completed INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                intent TEXT,
                name TEXT,
                email TEXT,
                company TEXT,
                phone TEXT,
                product TEXT,
                requirement TEXT,
                preferred_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


initialize_database()


# ==================================================
# INTENT DETECTION
# ==================================================

DEMO_PATTERNS = [
    r"\b(book|schedule|arrange|request|get|want|need|see|setup|set up)\b.{0,25}\bdemo\b",
    r"\bdemo\b.{0,15}\b(please|request)\b",
    r"\bproduct demo(nstration)?\b",
    r"\bdemonstration\b",
]


CONTACT_PATTERNS = [
    r"\b(contact|call|reach|email|ring)\s+me\b",
    r"\bcall\s*back\b|\bcallback\b",
    r"\bget in touch\b",
    r"\bfollow[\s-]?up\b.{0,20}\b(me|with me|call|email)\b",

    r"\b(talk|speak|connect)\b.{0,30}\b"
    r"(sales|someone|a person|human|representative|executive|team|expert|agent)\b",

    r"\b(contact|reach)\b.{0,12}\b(sales|your team|the team|support)\b",

    r"\b(speak|talk)\b.{0,20}\b(real|actual|live)\b.{0,15}\b(person|human|agent)\b",

    r"\bhow (can|do) i get started\b|\bget started\b.{0,15}\burgent",

    r"\b(have|get|want)\b.{0,15}\b(someone|your team|sales)\b.{0,15}\b(contact|call|reach)\b",

    r"\bi('| a)?m interested\b|\bi am interested\b|\binterested in (buying|purchasing|trying)\b",

    r"\b(sign me up|start a free trial|get a quote|request a quote)\b",
]


YES_WORDS = {
    "yes", "yeah", "yep", "sure", "ok", "okay",
    "please", "yes please", "y"
}

NO_WORDS = {
    "no", "nope", "nah", "no thanks", "not now", "n"
}

CANCEL_WORDS = {
    "cancel", "stop", "never mind", "nevermind",
    "exit", "no thanks"
}


FIELD_ORDER = ["name", "email", "phone"]


PROMPTS = {
    "name": "May I have your name, please?",
    "email": "Thanks{n}! What is your email address?",
    "phone": "And your phone number? (Type 'skip' if you prefer not to share.)",
}


def detect_lead_intent(message):

    m = message.lower()

    if re.search(
        r"not ready|not yet ready|just (looking|exploring)|maybe later",
        m
    ):
        return None

    if any(re.search(pattern, m) for pattern in DEMO_PATTERNS):
        return "demo"

    if any(re.search(pattern, m) for pattern in CONTACT_PATTERNS):
        return "sales"

    return None


# ==================================================
# FIELD PARSING
# ==================================================

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

PHONE_RE = re.compile(
    r"\+?\d[\d\s().-]{8,}\d"
)


def clean_name(text):

    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)

    text = re.sub(
        r"(?i)\b(my name is|name is|this is|call me|name:)\b",
        " ",
        text
    )

    text = re.sub(r"[^A-Za-z .'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-")

    words = text.split()

    # A valid name should contain 2-3 words at most.
    if not words or len(words) > 3:
        return ""

    # Reject single-character responses such as "I".
    if any(len(word) < 1 for word in words):
        return ""

    # Reject common sentence words.
    invalid_words = {
        "i", "want", "need", "speak", "talk", "connect",
        "real", "person", "human", "chatbot", "instead",
        "please", "contact", "team", "sales", "with",
        "the", "and", "for", "can", "you", "your",
        "how", "what", "why", "who", "when", "where",
        "my", "our", "we", "they", "do", "does",
        "is", "are", "will", "would", "could"
    }

    if any(word.lower() in invalid_words for word in words):
        return ""

    return text.title()


def looks_like_question(text):

    t = text.strip().lower()

    return t.endswith("?") or bool(
        re.match(
            r"(what|how|why|which|who|when|where|can|does|do|is|are|will|could)\b",
            t
        )
    )


def looks_like_sentence(text):

    words = text.strip().split()

    # A name should not be a long sentence.
    if len(words) > 3:
        return True

    sentence_words = {
        "want", "need", "speak", "talk", "connect",
        "because", "should", "would", "could",
        "please", "help", "explain", "solve",
        "access", "generate", "create", "provide"
    }

    return any(
        word.lower().strip(".,!?") in sentence_words
        for word in words
    )


def extract_details(message):

    """Extract email and phone from a message."""

    details = {
        "name": "",
        "email": "",
        "phone": ""
    }

    email_match = EMAIL_RE.search(message)

    if email_match:
        details["email"] = email_match.group(0)

    remaining_message = message.replace(
        details["email"], ""
    )

    phone_match = PHONE_RE.search(remaining_message)

    if phone_match:
        digits = re.sub(r"\D", "", phone_match.group(0))

        if len(digits) >= 10:
            details["phone"] = phone_match.group(0).strip()

    return details


# ==================================================
# DATABASE HELPERS
# ==================================================

def get_conversation(conversation_id):

    with sqlite3.connect(DB_PATH) as conn:

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        )

        row = cursor.fetchone()

        if row:
            return dict(row)

    return None


def save_conversation(
    conversation_id,
    intent,
    details,
    current_step,
    completed=0
):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT OR REPLACE INTO conversations
            (conversation_id, intent, details, current_step, completed)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conversation_id,
            intent,
            json.dumps(details),
            current_step,
            completed
        ))

        conn.commit()


def save_lead(conversation_id, intent, details):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT INTO leads
            (conversation_id, intent, name, email, company,
             phone, product, requirement, preferred_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            intent,
            details.get("name", ""),
            details.get("email", ""),
            details.get("company", ""),
            details.get("phone", ""),
            details.get("product", ""),
            details.get("requirement", ""),
            details.get("preferred_date", "")
        ))

        conn.commit()


# ==================================================
# STATE HELPERS
# ==================================================

def _load(conversation_id):

    conv = get_conversation(conversation_id) or {}

    if not conv:
        return conv, {}

    try:
        details = json.loads(conv["details"] or "{}")

    except (TypeError, json.JSONDecodeError):
        details = {}

    return conv, details


def next_missing(details):

    for field in FIELD_ORDER:

        if not details.get(field):
            return field

    return None


def prompt_for(field, details):

    name = (
        f" {details['name'].split()[0]}"
        if details.get("name")
        else ""
    )

    return PROMPTS[field].format(n=name)


def start_collection(conversation_id, intent, details=None):

    details = details or {}

    save_conversation(
        conversation_id,
        intent,
        details,
        "collect_details",
        0
    )

    field = next_missing(details) or "name"

    return prompt_for(field, details)


def offer_followup(conversation_id):

    """Ask whether the user wants a human follow-up."""

    save_conversation(
        conversation_id,
        "followup",
        {},
        "offer",
        0
    )

    return (
        "Would you like our team to follow up with you on this? "
        "Reply 'yes' and I'll take a few details."
    )


def pending_prompt(conversation_id):

    """Return the next missing field prompt if collection is active."""

    conv, details = _load(conversation_id)

    if (
        conv
        and not conv.get("completed")
        and conv.get("current_step") == "collect_details"
    ):

        field = next_missing(details)

        if field:
            return prompt_for(field, details)

    return None


def finish(conversation_id, conv_intent, details):

    save_lead(
        conversation_id,
        conv_intent,
        details
    )

    save_conversation(
        conversation_id,
        conv_intent,
        details,
        "completed",
        1
    )

    return (
        f"Thank you {details['name']}, "
        "our team will contact you shortly."
    )


# ==================================================
# MAIN ENTRY
# Returns (answer, handled)
# ==================================================

def process_lead_message(conversation_id, message):

    message = message.strip()

    if not message:
        return None, False

    low = message.lower().strip(" .!")

    conv, details = _load(conversation_id)

    active = bool(conv) and not conv.get("completed")

    # ----------------------------------------------
    # A. FOLLOW-UP OFFER
    # ----------------------------------------------

    if active and conv["current_step"] == "offer":

        if low in YES_WORDS or low.startswith(
            ("yes", "sure", "ok")
        ):

            return (
                "Sure! " + start_collection(
                    conversation_id,
                    "followup"
                ),
                True
            )

        # User declined or moved to another question.
        save_conversation(
            conversation_id,
            "followup",
            {},
            "declined",
            1
        )

        return None, False

    # ----------------------------------------------
    # B. COLLECTING LEAD DETAILS
    # ----------------------------------------------

    if active and conv["current_step"] == "collect_details":

        if low in CANCEL_WORDS or low in NO_WORDS:

            save_conversation(
                conversation_id,
                conv["intent"],
                details,
                "cancelled",
                1
            )

            return (
                "No problem! Let me know if you need anything else.",
                True
            )

        field = next_missing(details)

        found = extract_details(message)

        # Save email and phone if provided.
        for key in ("email", "phone"):

            if found[key] and not details.get(key):
                details[key] = found[key]

        # ------------------------------------------
        # NAME COLLECTION
        # ------------------------------------------

        if field == "name" and not details.get("name"):

            # Let normal questions go to the chatbot.
            if looks_like_question(message) and not found["email"]:
                return None, False

            # Do not treat sentences as names.
            if looks_like_sentence(message):
                return None, False

            name = clean_name(message)

            if not name:

                return (
                    "Sorry, I didn't catch your name. "
                    "Could you please enter your name?",
                    True
                )

            details["name"] = name

        # ------------------------------------------
        # EMAIL COLLECTION
        # ------------------------------------------

        elif field == "email" and not details.get("email"):

            if looks_like_question(message):
                return None, False

            return (
                "That doesn't look like a valid email. "
                "Could you re-enter it?",
                True
            )

        # ------------------------------------------
        # PHONE COLLECTION
        # ------------------------------------------

        elif field == "phone" and not details.get("phone"):

            if low == "skip":

                details["phone"] = "not provided"

            elif looks_like_question(message):

                return None, False

            else:

                return (
                    "Please enter a valid phone number "
                    "(10+ digits), or type 'skip'.",
                    True
                )

        # ------------------------------------------
        # NEXT FIELD OR FINISH
        # ------------------------------------------

        next_field = next_missing(details)

        if next_field:

            save_conversation(
                conversation_id,
                conv["intent"],
                details,
                "collect_details",
                0
            )

            return prompt_for(next_field, details), True

        return finish(
            conversation_id,
            conv["intent"],
            details
        ), True

    # ----------------------------------------------
    # C. NEW DEMO OR CONTACT REQUEST
    # ----------------------------------------------

    intent = detect_lead_intent(message)

    if intent:

        found = extract_details(message)

        details = {
            key: value
            for key, value in found.items()
            if value
        }

        lead = start_collection(
            conversation_id,
            intent,
            details
        )

        if intent == "demo":
            lead = "Happy to arrange a demo! " + lead

        else:
            lead = "Sure! " + lead

        return lead, True

    # ----------------------------------------------
    # D. NORMAL CHATBOT QUESTION
    # ----------------------------------------------

    return None, False
