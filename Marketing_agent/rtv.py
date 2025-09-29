from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

#GOOGLE_API_KEY
load_dotenv()
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)

# Loading vector DB
vectordb = FAISS.load_local(
    "./vectordb",
    embedding,
    allow_dangerous_deserialization=True
)

# Custom Prompt
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use only the information from the context below to answer the question.
Do not repeat the same information in your answer. If a fact appears in multiple places, mention it only once.
                                             
Context:
{context}

Question: {question}
Answer:""")

# Retriveal QA chain 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# Questionaire
query = "Cotes de Blaye"
result = qa_chain(query)

print("\n✅ Answer:\n", result["result"])

# showing from which file data comes
# print("\n📄 Source Files:")
for doc in result["source_documents"]:
    print(" -", doc.metadata.get("source"))
