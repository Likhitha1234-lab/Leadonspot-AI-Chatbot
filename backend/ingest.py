import json
import re
import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SCRAPED_FILE = (
    BASE_DIR / "scraped_data" / "leadonspot_pages.jsonl"
)

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "leadonspot_kb"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_TYPE = "leadonspot_website"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


# --------------------------------------------------
# LOAD SCRAPED PAGES
# --------------------------------------------------

def load_scraped_pages():
    if not SCRAPED_FILE.exists():
        raise FileNotFoundError(
            f"Scraped file not found: {SCRAPED_FILE}\n"
            "Run python scraper.py first."
        )

    documents = []

    with SCRAPED_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                page = json.loads(line)

                content = page.get("content", "").strip()
                url = page.get("url", "").strip()
                title = page.get("title", "").strip()

                if not content or not url:
                    continue

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "title": title,
                            "source_type": SOURCE_TYPE,
                        },
                    )
                )

            except json.JSONDecodeError:
                logging.warning(
                    "Skipping invalid JSON on line %s",
                    line_number,
                )

    logging.info(
        "Loaded %s website pages",
        len(documents),
    )

    return documents


# --------------------------------------------------
# LOAD KNOWLEDGE BASE TEXT FILES (faqs, products, recommendations...)
# --------------------------------------------------

KB_DIR = BASE_DIR.parent / "knowledge_base"
KB_SOURCE_TYPE = "knowledge_base"


def load_knowledge_base_files():
    documents = []

    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        # Split FAQ-style files per "Q:" and others per "## " heading,
        # so each chunk is one complete question/section.
        if "\nQ:" in text:
            parts = re.split(r"\n(?=Q:)", text)
        else:
            parts = re.split(r"\n(?=#{2,3} )", text)

        for part in parts:
            part = part.strip()
            if len(part) > 60:
                documents.append(Document(
                    page_content=part,
                    metadata={"source": path.name,
                              "title": path.stem,
                              "source_type": KB_SOURCE_TYPE},
                ))

    logging.info("Loaded %s knowledge-base sections", len(documents))
    return documents


# --------------------------------------------------
# SPLIT DOCUMENTS
# --------------------------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    logging.info(
        "Created %s text chunks",
        len(chunks),
    )

    return chunks


# --------------------------------------------------
# GENERATE STABLE CHUNK IDS
# --------------------------------------------------

def generate_chunk_id(document, index):
    raw = (
        document.metadata.get("source", "")
        + str(index)
        + document.page_content
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# INITIALIZE EMBEDDINGS
# --------------------------------------------------

def create_embeddings():
    logging.info("Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# --------------------------------------------------
# UPDATE WEBSITE RECORDS
# --------------------------------------------------

def update_website_records(vectorstore, chunks, source_type=SOURCE_TYPE):
    # Remove only records created by this website scraper.
    # Existing FAQ/product/pricing records remain untouched.

    try:
        existing = vectorstore.get(
            where={
                "source_type": source_type
            },
            include=[],
        )

        old_ids = existing.get("ids", [])

        if old_ids:
            logging.info(
                "Removing %s old website chunks...",
                len(old_ids),
            )

            vectorstore.delete(ids=old_ids)

    except Exception as error:
        raise RuntimeError(
            "Could not safely identify old website records. "
            "No new records were added. Check your ChromaDB "
            f"collection and metadata. Details: {error}"
        )

    if not chunks:
        logging.warning(
            "No website chunks found to ingest."
        )
        return

    ids = [
        generate_chunk_id(document, index)
        for index, document in enumerate(chunks)
    ]

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        vectorstore.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end],
        )

        logging.info(
            "Stored chunks %s to %s",
            start + 1,
            min(end, len(chunks)),
        )


# --------------------------------------------------
# MAIN INGESTION PIPELINE
# --------------------------------------------------

def ingest_website():
    logging.info("=" * 50)
    logging.info("LEADONSPOT WEBSITE INGESTION")
    logging.info("=" * 50)

    pages = load_scraped_pages()

    if not pages:
        raise RuntimeError(
            "No scraped pages found. "
            "Check scraper.py output before ingestion."
        )

    chunks = split_documents(pages)

    kb_chunks = split_documents(load_knowledge_base_files())

    embeddings = create_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    update_website_records(vectorstore, chunks)

    update_website_records(
        vectorstore,
        kb_chunks,
        source_type=KB_SOURCE_TYPE,
    )

    logging.info("=" * 50)
    logging.info("INGESTION COMPLETED")
    logging.info("Pages loaded: %s", len(pages))
    logging.info("Chunks created: %s", len(chunks))
    logging.info("Collection: %s", COLLECTION_NAME)
    logging.info("Database: %s", CHROMA_DIR)
    logging.info("=" * 50)


if __name__ == "__main__":
    ingest_website()