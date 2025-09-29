import os
import re
import json
import requests
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

VECTOR_DB_DIR = "./vectordb"
OUTPUT_FILE = "./news/filtered_news.json"
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # must be in .env


class TrendFetcher:
    def __init__(self, model="models/gemini-2.0-flash", embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(
            VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def build_queries(self, icp_json: dict) -> list:
        """Generate exactly 1 concise queries using LLM"""
        industry = icp_json.get("industry", "")
        pain_points = ", ".join(icp_json.get("customer_pain_points", []))
        needs = ", ".join(icp_json.get("customer_needs", []))

        prompt = f"""
        Generate exactly 1 concise search query (2–3 words) 
        about {industry} trends, strategy, or customer challenges. 

        Rules:
        - Each query must contain either "{industry}", "{pain_points}" or "{needs}".
        - Keep query short and distinct (no commas, no filler).
        - Do not copy phrases directly; rephrase if needed.
        - Output must be ONLY a valid JSON list of 1 string.

        Example:
        ["B2B SaaS GTM trends", "SaaS churn issues", "AI in SaaS", "SaaS CXO strategy", "SaaS growth 2025"]
        """

        response = self.llm.invoke(prompt).content.strip()
        match = re.search(r"\[.*\]", response, re.S)
        queries = json.loads(match.group()) if match else [f"{industry} trends"]

        cleaned = []
        for q in queries:
            q = re.sub(r"[^a-zA-Z0-9\s]", "", q).strip()
            words = q.split()
            if 2 <= len(words) <= 4:
                cleaned.append(" ".join(words))

        return cleaned if cleaned else [f"{industry} trends"]

    def fetch_serpapi(self, query: str, source: str = "google_news", num: int = 5) -> dict:
        """Fetch raw results from SerpAPI"""
        url = "https://serpapi.com/search"
        params = {"q": query, "engine": source, "num": num, "api_key": SERPAPI_KEY}
        resp = requests.get(url, params=params, timeout=20)
        return resp.json() if resp.status_code == 200 else {}

    def parse_results(self, data: dict, source: str) -> list:
        """Normalize SerpAPI results into a common schema"""
        results = []
        if not isinstance(data, dict):
            return results

        if source == "google_news":
            for art in data.get("news_results", []):
                results.append({
                    "title": art.get("title"),
                    "url": art.get("link"),
                    "description": art.get("snippet"),
                    "publishedAt": art.get("date"),
                    "source": "Google News"
                })

        elif source == "reddit":
            for post in data.get("organic_results", []):
                results.append({
                    "title": post.get("title"),
                    "url": post.get("link"),
                    "description": post.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "Reddit"
                })

        elif source == "twitter":
            for tweet in data.get("organic_results", []):
                results.append({
                    "title": tweet.get("title"),
                    "url": tweet.get("link"),
                    "description": tweet.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "Twitter"
                })

        elif source == "linkedin":
            for post in data.get("organic_results", []):
                results.append({
                    "title": post.get("title"),
                    "url": post.get("link"),
                    "description": post.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "LinkedIn"
                })

        elif source == "youtube":
            for vid in data.get("video_results", []):
                results.append({
                    "title": vid.get("title"),
                    "url": vid.get("link"),
                    "description": vid.get("snippet"),
                    "publishedAt": vid.get("date", datetime.utcnow().isoformat()),
                    "source": "YouTube"
                })

        return results

    def relevance_filter(self, items: list, icp_json: dict, top_k: int = 10, threshold: float = 0.65) -> list:
        """Filter results using ICP keywords + vector similarity with threshold"""
        keywords = [icp_json.get("industry", "").lower()]
        keywords += [kw.lower() for kw in icp_json.get("customer_pain_points", [])]
        keywords += [kw.lower() for kw in icp_json.get("customer_needs", [])]

        filtered = []
        for item in items:
            text = f"{item.get('title','')} {item.get('description','')}".lower()

            keyword_hit = any(kw in text for kw in keywords if kw)
            docs_and_scores = self.vectordb.similarity_search_with_score(text, k=1)
            semantic_hit = docs_and_scores and docs_and_scores[0][1] >= threshold

            if keyword_hit or semantic_hit:
                filtered.append({
                    **item,
                    "relevance_context": docs_and_scores[0][0].page_content if docs_and_scores else "",
                    "similarity_score": float(docs_and_scores[0][1]) if docs_and_scores else None,  # ✅ cast
                })

        # ✅ keep only top_k
        return filtered[:top_k]

    def run(self, icp_json_path="./niche/niche_icp.json"):
        with open(icp_json_path, "r") as f:
            icp_json = json.load(f)

        queries = self.build_queries(icp_json)
        print(f"🔍 Queries: {queries}")

        all_items = []
        for q in queries:
            raw = self.fetch_serpapi(q, source="google_news")
            print(f"📥 google_news returned {len(raw) if isinstance(raw, dict) else 0} keys for query '{q}'")
            items = self.parse_results(raw, source="google_news")
            print(f"   Parsed {len(items)} items from google_news")
            all_items.extend(items)

        # ✅ only top 10 will be returned
        filtered = self.relevance_filter(all_items, icp_json, top_k=10)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=4)

        print(f"✅ Saved {len(filtered)} relevant trends to {OUTPUT_FILE}")
        return filtered


if __name__ == "__main__":
    fetcher = TrendFetcher()
    results = fetcher.run()
    for r in results:
        print(f"- [{r['source']}] {r['title']} ({r['url']})")
