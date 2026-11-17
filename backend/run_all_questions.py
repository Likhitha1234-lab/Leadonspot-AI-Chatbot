"""Runs all 231 test questions through the real chatbot and saves answers to
test_results.csv so you can review them. Needs GROQ_API_KEY and chroma_db.
Each question is asked in a FRESH conversation, except the memory section
which is asked after a short setup chat.
Usage: python run_all_questions.py"""
import csv
from chatbot import get_chatbot_response, UNKNOWN_ANSWER
from route_check import route
from test_questions import SECTIONS

SETUP = [
    ("My company is in logistics and I have 15 salespeople.",
     "Leadonspot Smart CRM and Nexus can help logistics teams manage leads and conversations."),
]

with open("test_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["section", "question", "route", "answer", "is_unknown"])
    for section, qs in SECTIONS.items():
        for q in qs:
            history = []
            if "Memory" in section:
                for u, a in SETUP:
                    history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            try:
                ans = get_chatbot_response(q, history)
            except Exception as e:
                ans = f"ERROR: {e}"
            w.writerow([section, q, route(q), ans, ans.startswith(UNKNOWN_ANSWER[:40])])
            print(f"[{section}] {q}\n   -> {ans}\n")
print("Saved test_results.csv")