
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
# LOAD KNOWLEDGE BASE TEXT FILES
# --------------------------------------------------

KB_DIR = BASE_DIR.parent / "knowledge_base"
KB_SOURCE_TYPE = "knowledge_base"


def load_knowledge_base_files():

    documents = []

    for path in sorted(KB_DIR.glob("*.txt")):

        text = path.read_text(encoding="utf-8")

        if "\nQ:" in text:
            parts = re.split(r"\n(?=Q:)", text)
        else:
            parts = re.split(r"\n(?=#{2,3} )", text)

        for part in parts:

            part = part.strip()

            if len(part) > 60:

                documents.append(
                    Document(
                        page_content=part,
                        metadata={
                            "source": path.name,
                            "title": path.stem,
                            "source_type": KB_SOURCE_TYPE,
                        },
                    )
                )

    logging.info(
        "Loaded %s knowledge-base sections",
        len(documents),
    )

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

def update_website_records(
    vectorstore,
    chunks,
    source_type=SOURCE_TYPE,
):

    # Remove only records created by this source type.
    # Other records remain untouched.

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
                "Removing %s old records...",
                len(old_ids),
            )

            vectorstore.delete(ids=old_ids)

    except Exception as error:

        raise RuntimeError(
            "Could not safely identify old records. "
            "No new records were added. Check your ChromaDB "
            f"collection and metadata. Details: {error}"
        )

    if not chunks:

        logging.warning(
            "No chunks found to ingest."
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
# WORKFLOW AUTOMATION Q&A
# --------------------------------------------------

workflow_qa = [

    {
        "question": "Can I automatically assign new leads to sales representatives?",
        "answer": "Yes. Leadonspot supports automated lead assignment and routing through workflow rules."
    },

    {
        "question": "Can I send a welcome email when a new lead is created?",
        "answer": "Leadonspot supports automated email workflows and follow-up sequences. A specific welcome-email trigger needs confirmation."
    },

    {
        "question": "Can I create a follow-up task automatically?",
        "answer": "Yes. Leadonspot supports automatic follow-up task creation and reminders."
    },

    {
        "question": "Can I notify managers when a lead remains unattended?",
        "answer": "Leadonspot supports workflow triggers, reminders and notifications. A specific manager alert for unattended leads needs confirmation."
    },

    {
        "question": "Can I trigger automation when a deal gets stuck?",
        "answer": "Leadonspot AI Opportunity Health identifies deals at risk using response time, sentiment and stage velocity. Automatic workflow triggering needs confirmation."
    },

    {
        "question": "Can I assign leads using round-robin distribution?",
        "answer": "Leadonspot supports automated lead assignment, but round-robin distribution is not explicitly confirmed."
    },

    {
        "question": "Can I create different workflows for different industries?",
        "answer": "Leadonspot supports configurable workflow rules and industry-specific sales processes."
    },

    {
        "question": "Can I send notifications when a lead changes pipeline stage?",
        "answer": "Leadonspot supports pipeline automation and workflow triggers. The exact stage-change notification needs confirmation."
    },

    {
        "question": "Can I automate follow-ups after two days of inactivity?",
        "answer": "Leadonspot supports follow-up automation and reminders. A specific two-day inactivity trigger is not confirmed."
    },

    {
        "question": "Can I create multiple conditions in a single workflow?",
        "answer": "Leadonspot supports workflow configuration. The exact multiple-condition functionality needs confirmation."
    },

    {
        "question": "Can I stop an automation once a deal is closed?",
        "answer": "Leadonspot supports pipeline automation and configurable workflow rules. The exact stop condition needs confirmation."
    },

    {
        "question": "Can I automate lead assignment based on location?",
        "answer": "Leadonspot supports automated lead routing and assignment. Location-based assignment is not explicitly confirmed."
    },
    {
        "question": "Do you remember how many salespeople I have?",
        "answer": (
            "I don't have your team's salesperson count available "
            "in this conversation. If you share the number, I can "
            "use it while helping you with your sales-related questions."
        )
    },
    {
        "question": "Based on my requirements, what is my biggest business challenge?",
        "answer": (
            "Based on what you've shared, your biggest business "
            "challenge appears to be managing and following up with "
            "leads effectively. This can result in missed opportunities "
            "and slower conversions. If you share more details about "
            "your requirements, I can give you a more accurate assessment."
        )
    },
        
    {
        "question": "Can I get information about a company's recent news?",
        "answer": (
            "Company intelligence can help you understand a business "
            "through relevant company information, industry details, "
            "and business insights. For specific information about "
            "recent company news, please contact the Leadonspot team "
            "for confirmation."
        )
    }


]


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

    kb_chunks = split_documents(
        load_knowledge_base_files()
    )

    embeddings = create_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Existing website ingestion
    update_website_records(
        vectorstore,
        chunks,
    )

    # Existing knowledge-base ingestion
    update_website_records(
        vectorstore,
        kb_chunks,
        source_type=KB_SOURCE_TYPE,
    )

    # --------------------------------------------------
    # ADD WORKFLOW AUTOMATION Q&A
    # --------------------------------------------------

    workflow_documents = []

    workflow_ids = []

    for item in workflow_qa:

        content = (
            "Category: Workflow Automation Scenarios\n\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )

        document = Document(
            page_content=content,
            metadata={
                "source": "workflow_automation",
                "title": "Workflow Automation",
                "source_type": "manual_qa",
                "category": "workflow_automation",
            },
        )

        workflow_documents.append(document)

        record_id = hashlib.sha256(
            item["question"].encode("utf-8")
        ).hexdigest()

        workflow_ids.append(
            "workflow_automation_" + record_id
        )

    # Remove only previously added workflow records
    existing_workflow = vectorstore.get(
    where={
        "$and": [
            {"source_type": "manual_qa"},
            {"category": "workflow_automation"},
        ]
    },
    include=[],
)

    old_workflow_ids = existing_workflow.get("ids", [])

    if old_workflow_ids:

        vectorstore.delete(
            ids=old_workflow_ids
        )

        logging.info(
            "Removed %s old Workflow Automation records.",
            len(old_workflow_ids),
        )

    # Add the updated 12 records
    vectorstore.add_documents(
        documents=workflow_documents,
        ids=workflow_ids,
    )

    logging.info(
        "Added %s Workflow Automation Q&A records.",
        len(workflow_documents),
    )

    # --------------------------------------------------
    # INGESTION SUMMARY
    # --------------------------------------------------

    logging.info("=" * 50)
    logging.info("INGESTION COMPLETED")
    logging.info("Pages loaded: %s", len(pages))
    logging.info("Website chunks created: %s", len(chunks))
    logging.info("Knowledge-base chunks created: %s", len(kb_chunks))
    logging.info("Workflow Automation Q&A: %s", len(workflow_documents))
    logging.info("Collection: %s", COLLECTION_NAME)
    logging.info("Database: %s", CHROMA_DIR)
    logging.info("=" * 50)


if __name__ == "__main__":
    ingest_website()
