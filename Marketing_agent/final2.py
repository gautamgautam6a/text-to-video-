import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load env vars

load_dotenv()


# ================== CONFIG ==================
vectordb_dir = "./vectordb"
record_file = "embedding_record.json"
pdf_dir = "./data"

embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="mistral", temperature=0.1)

# ================== HELPERS ==================
def load_embedding_record():
    if Path(record_file).exists():
        with open(record_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"embedded_pdfs": []}

def save_embedding_record(record):
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

def load_vectorstore(vectordb_dir=vectordb_dir, pdf_dir=pdf_dir):
    index_dir = Path(vectordb_dir)
    record = load_embedding_record()

    if index_dir.exists() and any(index_dir.glob("*.faiss")):
        print("🟢 [DEBUG] Loading FAISS vector database...")
        vectordb = FAISS.load_local(vectordb_dir, embedding, allow_dangerous_deserialization=True)
        docs = vectordb.similarity_search("", k=200)
        pdf_files = list(set([d.metadata.get("source", "Unknown") for d in docs]))
        print(f"✅ [DEBUG] Loaded {len(docs)} documents from {len(pdf_files)} PDFs")
        return vectordb, docs, pdf_files

    print("🟡 [DEBUG] No FAISS index found, creating new one...")
    all_docs = []
    embedded_pdfs = record.get("embedded_pdfs", [])

    for pdf in Path(pdf_dir).glob("*.pdf"):
        if str(pdf) in embedded_pdfs:
            print(f"⏭️ [DEBUG] Skipping already embedded PDF: {pdf.name}")
            continue
        print(f"📥 [DEBUG] Loading PDF: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        all_docs.extend(docs)
        embedded_pdfs.append(str(pdf))

    if not all_docs:
        raise RuntimeError("❌ [DEBUG] No new PDFs found to embed.")

    vectordb = FAISS.from_documents(all_docs, embedding)
    vectordb.save_local(vectordb_dir)
    record["embedded_pdfs"] = embedded_pdfs
    save_embedding_record(record)

    pdf_files = list(set([d.metadata.get("source", "Unknown") for d in all_docs]))
    print(f"✅ [DEBUG] Created FAISS index with {len(all_docs)} docs from {len(pdf_files)} PDFs")
    return vectordb, all_docs, pdf_files

def summarize_pdfs(docs):
    print(f"🟡 [DEBUG] Starting summarization for {len(docs)} documents")
    summaries = {}
    for doc in docs:
        text = doc.page_content[:1000]
        prompt = f"Summarize this text in 5 bullet points:\n\n{text}"
        try:
            summary = llm.invoke(prompt)
            summaries[doc.metadata.get("source", "Unknown")] = summary
        except Exception as e:
            summaries[doc.metadata.get("source", "Unknown")] = f"Error: {e}"
    print(f"✅ [DEBUG] Completed summarization for {len(summaries)} PDFs")
    return summaries

def setup_qa(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    prompt_template = """
    Use the following context to answer the question.
    If the answer cannot be found, say "I don't know".

    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

# ================== MAIN ==================
if __name__ == "__main__":
    try:
        vectordb, docs, pdf_files = load_vectorstore()

        summaries = summarize_pdfs(docs)

        qa_chain = setup_qa(vectordb)

        sample_question = "What is the main idea of the PDFs?"
        answer = qa_chain.run(sample_question)

        final_state = {
            "selected_pdfs": pdf_files,
            "summaries": summaries,
            "sample_question": sample_question,
            "answer": answer,
        }

        print("\n📊 Final State:")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ [ERROR] {e}")
