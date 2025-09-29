import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END

# Load env vars
load_dotenv()

PDF_DIR = "./data"
VECTOR_DB_DIR = "./vectordb"
RECORD_FILE = "embedding_record.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embedding = OllamaEmbeddings(model="nomic-embed-text")

# ----------- STATE -------------
class State(dict):
    """Carries state across graph nodes"""
    record: dict
    new_files: list
    raw_docs: list
    docs: list
    vectordb: object
    updated_record: dict

# ----------- HELPERS -------------
def load_embedding_record(record_file):
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            return json.load(f)
    return {}

def save_embedding_record(record, record_file):
    with open(record_file, "w") as f:
        json.dump(record, f, indent=2)

# ----------- NODES -------------
def check_new_pdfs(state: State):
    """Identify new/changed PDFs"""
    print("\n🔍 Checking for new/changed PDFs...")
    new_files = []
    updated_record = state["record"].copy()

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            full_path = os.path.join(PDF_DIR, filename)
            last_modified = os.path.getmtime(full_path)
            if filename not in state["record"] or state["record"][filename] != last_modified:
                new_files.append(filename)
                updated_record[filename] = last_modified

    print(f"📂 Found {len(new_files)} new/changed PDFs: {new_files}" if new_files else "✅ No new PDFs.")
    return {"new_files": new_files, "updated_record": updated_record}

def load_pdfs(state: State):
    """Load all new PDFs"""
    print(f"\n📥 Loading PDFs: {state['new_files']}")
    all_docs = []
    for filename in state["new_files"]:
        full_path = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(full_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)
    print(f"📑 Loaded {len(all_docs)} pages from {len(state['new_files'])} PDFs")
    return {"raw_docs": all_docs}

def split_documents(state: State):
    """Split loaded docs into chunks"""
    print("\n✂ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(state["raw_docs"])
    print(f"🔖 Created {len(docs)} chunks from {len(state['raw_docs'])} pages")
    return {"docs": docs}

def update_vectordb(state: State):
    """Update or create FAISS index"""
    print("\n🗂 Updating FAISS vector store...")
    faiss_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    if os.path.exists(faiss_path):
        vectordb = FAISS.load_local(VECTOR_DB_DIR, embedding, allow_dangerous_deserialization=True)
        vectordb.add_documents(state["docs"])
        print(f"➕ Added {len(state['docs'])} chunks to existing FAISS index")
    else:
        vectordb = FAISS.from_documents(state["docs"], embedding)
        print(f"🆕 Created new FAISS index with {len(state['docs'])} chunks")
    vectordb.save_local(VECTOR_DB_DIR)
    return {"vectordb": vectordb}

def update_record(state: State):
    """Save updated embedding record"""
    print("\n💾 Saving embedding record...")
    save_embedding_record(state["updated_record"], RECORD_FILE)
    print("✅ Embedding record updated")
    return {}

def no_changes(state: State):
    """Handle the case of no new PDFs"""
    print("\n✅ No new PDFs to process. Vector DB already up to date.")
    return {}

# ----------- BUILD GRAPH -------------
workflow = StateGraph(State)

# Nodes
workflow.add_node("check_new_pdfs", check_new_pdfs)
workflow.add_node("load_pdfs", load_pdfs)
workflow.add_node("split", split_documents)
workflow.add_node("update_vectordb", update_vectordb)
workflow.add_node("update_record", update_record)
workflow.add_node("no_changes", no_changes)

# Entry point
workflow.set_entry_point("check_new_pdfs")

# Branching logic
def has_new_files(state: State):
    return "load_pdfs" if state["new_files"] else "no_changes"

workflow.add_conditional_edges("check_new_pdfs", has_new_files, {
    "load_pdfs": "load_pdfs",
    "no_changes": "no_changes",
})

# Sequential edges
workflow.add_edge("load_pdfs", "split")
workflow.add_edge("split", "update_vectordb")
workflow.add_edge("update_vectordb", "update_record")
workflow.add_edge("update_record", END)
workflow.add_edge("no_changes", END)

# Compile app
app = workflow.compile()

# ----------- RUN PIPELINE -------------
if __name__ == "__main__":
    print("🚀 Starting LangGraph PDF Embedding Workflow")
    initial_state = State(record=load_embedding_record(RECORD_FILE))
    final_state = app.invoke(initial_state)
    print("\n🎉 Workflow finished.")