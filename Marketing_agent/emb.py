import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import json

#GOOGLE_API_KEY
load_dotenv()

PDF_DIR = "./data"
VECTOR_DB_DIR = "./vectordb"
RECORD_FILE = "embedding_record.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

from langchain_community.embeddings import OllamaEmbeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")


#Gemini model------------------------
#embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Load or create the embedding record
def load_embedding_record(record_file):
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            return json.load(f)
    return {}

def save_embedding_record(record, record_file):
    with open(record_file, "w") as f:
        json.dump(record, f, indent=2)

# Loading of PDF (incremental) ----------------------------
def load_and_tag_new_pdfs(pdf_dir, record):
    all_docs = []
    updated_record = record.copy()
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(pdf_dir, filename)
            last_modified = os.path.getmtime(full_path)
            # Only process if new or changed
            if filename not in record or record[filename] != last_modified:
                print(f"📄 Loading (new/changed) {filename}...")
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = doc.metadata.get("page", None)
                all_docs.extend(docs)
                updated_record[filename] = last_modified
            else:
                print(f"✅ Skipping unchanged {filename}")
    return all_docs, updated_record

# Splitting doc in chunks------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# Creating vector db------------------
def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

# Creating vector db--------------------
def save_vector_store(vectorstore, path):
    if not os.path.exists(path):
        os.makedirs(path)
    vectorstore.save_local(path)

def main():
    print("\n📥 Loading embedding record...")
    record = load_embedding_record(RECORD_FILE)

    print("\n📥 Loading and tagging new/changed PDFs...")
    raw_docs, updated_record = load_and_tag_new_pdfs(PDF_DIR, record)

    if not raw_docs:
        print("\n✅ No new or changed PDFs to process. Vector DB is up to date!")
        return

    print("\n✂ Splitting into chunks...")
    docs = split_documents(raw_docs)

    # Check if FAISS index exists
    faiss_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    if os.path.exists(faiss_path):
        print("\n🔗 Loading existing FAISS vector store...")
        vectordb = FAISS.load_local(VECTOR_DB_DIR, embedding, allow_dangerous_deserialization=True)
        print("\n➕ Adding new/changed documents to vector store...")
        vectordb.add_documents(docs)
    else:
        print("\n🔗 Creating new FAISS vector store with metadata...")
        vectordb = create_vector_store(docs, embedding)

    print("\n💾 Saving vector DB to disk...")
    save_vector_store(vectordb, VECTOR_DB_DIR)

    print("\n💾 Updating embedding record...")
    save_embedding_record(updated_record, RECORD_FILE)

    print("\n✅ Vector DB updated with new/changed PDFs!")

if __name__ == "__main__":
    main()