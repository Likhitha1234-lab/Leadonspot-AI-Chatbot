
import sqlite3
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "leads.db"


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


def extract_details(message):
    details = {
        "name": "",
        "email": "",
        "company": "",
        "phone": "",
        "product": "",
        "requirement": ""
    }

    message = message.strip()

    # Extract email
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        message
    )

    if email_match:
        details["email"] = email_match.group(0)

    # Extract phone number
    phone_match = re.search(
        r"\+?\d[\d\s().-]{8,}\d",
        message
    )

    if phone_match:
        details["phone"] = phone_match.group(0).strip()

    # Handle comma-separated details:
    # Name, email, company
    parts = [part.strip() for part in message.split(",") if part.strip()]

    if details["email"] and len(parts) >= 3:
        email_index = next(
            (i for i, part in enumerate(parts) if details["email"] in part),
            -1
        )

        if email_index > 0:
            details["name"] = parts[email_index - 1]

        if email_index >= 0 and email_index + 1 < len(parts):
            details["company"] = parts[email_index + 1]

    # Handle details written as:
    # Likhitha N from ABC Technologies
    if not details["company"]:
        company_match = re.search(
            r"\b(?:from|at|working at|company is)\s+(.+)$",
            message,
            re.IGNORECASE
        )

        if company_match:
            details["company"] = company_match.group(1).strip(" ,.")

            name_part = message[:company_match.start()].strip(" ,.")
            if not details["name"]:
                details["name"] = name_part

    # Remove email and phone from extracted name
    if details["name"]:
        details["name"] = details["name"].strip(" ,.")

    return details


def detect_lead_intent(message):
    message = message.lower()

    demo_keywords = [
        "product demo",
        "schedule a demo",
        "schedule demo",
        "book a demo",
        "book demo",
        "arrange a demo",
        "request a demo",
        "demo please"
    ]

    sales_keywords = [
        "contact sales",
        "talk to sales",
        "speak with sales",
        "sales team",
        "sales representative"
    ]

    if any(keyword in message for keyword in demo_keywords):
        return "demo"

    if any(keyword in message for keyword in sales_keywords):
        return "sales"

    return None


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


def save_conversation(conversation_id, intent, details, current_step, completed=0):
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


def process_lead_message(conversation_id, message):
    message = message.strip()

    if not message:
        return None, False

    conversation = get_conversation(conversation_id)
    extracted = extract_details(message)

    # CASE 1: User provides all contact details together.
    # This works even if no lead-capture conversation was started.
    if (
        extracted["name"]
        and extracted["email"]
        and extracted["company"]
    ):
        details = extracted.copy()

        intent = "demo"

        if conversation:
            try:
                previous_details = json.loads(conversation["details"] or "{}")
                previous_details.update({
                    key: value
                    for key, value in extracted.items()
                    if value
                })
                details = previous_details
                intent = conversation["intent"] or "demo"
            except (TypeError, json.JSONDecodeError):
                pass

        save_lead(conversation_id, intent, details)

        save_conversation(
            conversation_id,
            intent,
            details,
            "completed",
            1
        )

        return (
            f"Thank you for your details, {details['name']}. "
            "Our team will contact you shortly."
        ), True

    # CASE 2: Continue an existing lead-capture conversation.
    if conversation:
        if conversation["completed"]:
            return None, False

        try:
            details = json.loads(conversation["details"] or "{}")
        except (TypeError, json.JSONDecodeError):
            details = {}

        details.update({
            key: value
            for key, value in extracted.items()
            if value
        })

        if not details.get("name"):
            return "Please share your full name.", True

        if not details.get("email"):
            return "Please share your email address.", True

        if not details.get("company"):
            return "Please share your company name.", True

        save_lead(conversation_id, conversation["intent"], details)

        save_conversation(
            conversation_id,
            conversation["intent"],
            details,
            "completed",
            1
        )

        return (
            f"Thank you for your details, {details['name']}. "
            "Our team will contact you shortly."
        ), True

    # CASE 3: Start lead capture when user requests a demo or sales contact.
    intent = detect_lead_intent(message)

    if intent:
        save_conversation(
            conversation_id,
            intent,
            {},
            "collect_details",
            0
        )

        return (
            "Yes, we can arrange a product demo. "
            "Please share your name, email address, and company "
            "so we can set it up."
        ), True

    return None, False
