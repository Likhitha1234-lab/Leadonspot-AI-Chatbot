
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="leadonspot_kb",
    persist_directory=str(BASE_DIR / "chroma_db"),
    embedding_function=embeddings,
)

questions = [
    "Can I manage the complete sales lifecycle?",
    "Can I automatically assign new leads to sales representatives?",
    "Can I detect duplicate leads?",
    "Can I integrate Google Sheets?",
    "Can I find decision-makers by designation?",
    "Can the AI agent qualify leads during calls?",
    "Can I manage multiple sales teams?",
    "Can I export my data if I leave Leadonspot?",
    "How does Leadonspot handle customer data security?",
    "Can I automate follow-ups after two days of inactivity?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    results = db.similarity_search_with_relevance_scores(
        question,
        k=3,
        filter={"source_type": "leadonspot_website"},
    )

    for document, score in results:
        print(f"\nRelevance score: {score:.3f}")
        print("Title:", document.metadata.get("title"))
        print("Source:", document.metadata.get("source"))
        print("Content:", document.page_content[:500])
