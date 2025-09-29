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

Hard rules:
- Do NOT copy sentences or phrases verbatim from the input. Rephrase and summarize.
- Produce fact-based, granular, and actionable reasons for problems ("why") and concrete mechanisms for solutions ("how").
- If a fact cannot be inferred, leave it as "" for strings or [] for lists.
- Output ONLY valid JSON matching the schema below. No extra text, no commentary.

From the provided context, extract and rephrase the following fields.

Schema & required content expectations:

- industry: (string) concise category, e.g., "B2B SaaS Go-To-Market"
- target_audience: (list of strings) concise roles or groups, e.g., ["Founders", "RevOps"]
- customer_pain_points: (list of objects) each object MUST include:
    - challenge: (string) short rephrased challenge
    - why: (list of objects) each object is a specific possible cause with:
        - cause: (string) concise label of the cause (e.g., "poor data instrumentation")
        - explanation: (string) 1-3 sentences explaining the causal mechanism — how the cause produces the challenge (be specific; cite typical mechanisms or system failures)
        - indicators: (list of strings) concrete signals, metrics, or behaviors you would observe if this cause is true (e.g., "conversion drops at demo stage", "CRM lead source empty")
        - recommended_first_checks: (list of short actions) immediate, low-effort checks or data points to verify this cause (e.g., "review last 30 CRM records for missing source")
- customer_needs: (list of objects) each object MUST include:
    - need: (string) desired outcome
    - how: (list of objects) each object is a concrete way the company can meet that need:
        - approach: (string) short name of the approach/mechanism (e.g., "instrumentation + analytics layer")
        - details: (string) 1-3 sentences describing exactly how it works or is implemented (tools, integration points, behaviors)
        - measurable_signs: (list of strings) KPIs or signals that show the approach is working (e.g., "win rate +7pp", "reduction in average sales cycle")
        - first_deliverables: (list of strings) immediate outputs to expect (e.g., "dashboard of 5 leading GTM metrics", "20 prioritized playbook changes")
- value_proposition: (string) 1 sentence paraphrase of unique offering
- brand_tone: (string) inferred communication style, e.g., "Strategic, Direct, Thought-leadership"

Output format (must match exactly):
{{
  \"industry": \"\",
  \"target_audience\": [],
  \"customer_pain_points\": [
    {{
      \"challenge": \"\",
      \"why\": [
        {{
          \"cause\": \"\",
          \"explanation\": \"\",
          \"indicators\": [],
          \"recommended_first_checks\": [],
        }}
      ]
    }}
  ],
  \"customer_needs\": [
    {{
      \"need\": \"\",
      \"how\": [
        {{
          \"approach\": \"\",
          \"details\": \"\",
          \"measurable_signs\": [],
          \"first_deliverables\": []
        }}
      ]
    }}
  ],
  \"value_proposition\": \"\",
  \"brand_tone\": \"\"
}}

Context:
{context}
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
