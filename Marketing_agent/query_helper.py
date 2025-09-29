import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = "./vectordb"

class ICPQueryHelper:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(
            VECTOR_DB_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def query(self, question: str, k: int = 5) -> str:
        """Ask a question against ICP/Niche knowledge base"""
        docs = self.vectordb.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an assistant for analyzing a company's business niche and ICP documents.
        Based on the context below, answer the question clearly.

        Context:
        {context}

        Question:
        {question}
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

if __name__ == "__main__":
    helper = ICPQueryHelper()
    while True:
        q = input("\n❓ Ask ICP Knowledge Base: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = helper.query(q)
        print(f"\n💡 Answer: {answer}")
