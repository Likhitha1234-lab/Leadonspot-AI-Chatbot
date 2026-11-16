
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge_base"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"

COLLECTION_NAME = "leadonspot_kb"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_knowledge_base():

    # 1. Load documents
    loader = DirectoryLoader(
        str(KB_PATH),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No TXT files found in knowledge_base folder!")

    print(f"Loaded {len(documents)} documents")

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Generate embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    # 4. Remove previous database for fresh ingestion
    if DB_PATH.exists():
        import shutil
        shutil.rmtree(DB_PATH)

    # 5. Store documents in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
        collection_name=COLLECTION_NAME
    )

    print("Knowledge base successfully created!")
    print("Total records:", vectorstore._collection.count())
    print("Database location:", DB_PATH)


if __name__ == "__main__":
    create_knowledge_base()
