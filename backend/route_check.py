"""Offline check: which rule handles each test question (no Groq / Chroma needed)."""
import importlib.util
import os, sys, types, re
from pathlib import Path

# Dummy key ONLY when there is no real .env (never hide the real key).
if not (Path(__file__).resolve().parent / ".env").exists():
    os.environ.setdefault("GROQ_API_KEY", "offline-test")
class _C:                                    # fake Chroma
    def __init__(self, **k):
        self._collection = types.SimpleNamespace(count=lambda: 1)


def _stub_if_missing(name, **attrs):
    """Fake a library ONLY when it is not installed. Never replace a real
    one, otherwise the real Groq client breaks when this file is imported."""
    if name in sys.modules:
        return
    try:
        if importlib.util.find_spec(name) is not None:
            return
    except (ImportError, ValueError):
        pass
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


_stub_if_missing("groq", Groq=lambda **k: None)
_stub_if_missing("dotenv", load_dotenv=lambda *a, **k: None)
_stub_if_missing("langchain_huggingface", HuggingFaceEmbeddings=lambda **k: None)
_stub_if_missing("langchain_chroma", Chroma=_C)
import chatbot as c
import lead_capture as lc

def route(q):
    if re.search(r"previous (conversation|session|chat)|last (time|session|chat)|yesterday|where we (left|stopped)", q.lower()): return "SESSION-NOTE"
    if c.handle_sensitive_question(q): return "PRIVACY"
    if c.handle_risky_question(q): return "RISKY-RULE"
    if c.is_out_of_scope_question(q): return "OUT-OF-SCOPE"
    if c.is_pricing_question(q) and c.is_future_pricing_question(q): return "FUTURE-PRICE"
    if lc.detect_lead_intent(q): return "LEAD-CAPTURE"
    if c.is_pricing_question(q): return "PRICING-FILE"
    return "RAG"

if __name__ == "__main__":
    import test_questions as tq
    for section, qs in tq.SECTIONS.items():
        print(f"\n## {section}")
        for q in qs:
            print(f"  {route(q):<13} {q}")