import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()

VECTOR_DB_DIR = "./vectordb"
OUTPUT_FILE = "./niche/niche_icp.json"

# ----------- PROMPT -------------
PROMPT_TEMPLATE = """
You are an AI assistant that extracts structured business insights
from documents about a company's niche and Ideal Customer Profile (ICP).

Your task is NOT to copy exact words or phrases from the text.
Instead, interpret the meaning, rephrase in your own words,
and generalize where possible.

If the document does not state something explicitly, infer it logically
from the surrounding context or leave it as an empty string/list.

Based on the provided context, extract and rephrase the following fields:

- industry: (string, a concise category, not a full sentence)
- target_audience: (list of concise roles or groups, e.g., ["CXOs", "Sales Heads"])
- customer_pain_points: (list of short, rephrased challenges customers face and sub-points detailing "why?" such challenges exist)
- customer_needs: (list of short, rephrased desired outcomes or solutions, and sub-points detailing "How our company exists in this?")
- value_proposition: (string, one clear, paraphrased summary sentence)
- brand_tone: (string, inferred writing/communication style, e.g., Professional, Thought-leadership, Conversational)

Context:
{context}

Respond ONLY in JSON format with the following keys:
industry, target_audience, customer_pain_points, customer_needs, value_proposition, brand_tone
"""


# ----------- FUNCTIONS -------------
def load_faiss_index():
    """Load existing FAISS index"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectordb

def query_icp(vectordb, query="Summarize the niche and ICP details"):
    """Query FAISS index for ICP context"""
    docs = vectordb.similarity_search(query, k=10)
    return "\n\n".join([doc.page_content for doc in docs])

def extract_structured_info(context: str, model="models/gemini-2.5-flash") -> dict:
    """Use LLM to extract structured JSON from context"""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    prompt = PromptTemplate(input_variables=["context"], template=PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=JsonOutputParser())
    response = chain.run({"context": context})
    
    # Parse into dict if string
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except:
            raise ValueError("⚠️ LLM did not return valid JSON.")
    return response

def save_json(data: dict, output_file: str = OUTPUT_FILE):
    """Save extracted JSON"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved structured info to {output_file}")

# ----------- MAIN -------------
if __name__ == "__main__":
    print("🚀 Generating niche_icp.json from embeddings...")
    vectordb = load_faiss_index()
    context = query_icp(vectordb)
    structured_data = extract_structured_info(context)
    save_json(structured_data)
    print("🎉 niche_icp.json created successfully.")
