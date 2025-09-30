import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

TOPICS_FILE = "./topics/topics.json"
NEWS_FILE = "./news/filtered_news.json"
NICHE_FILE = "./niche/niche_icp.json"
OUTPUT_FILE = "./content/generated_content.json"
VECTOR_DB_DIR = "./vectordb"


class ContentGenerator:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)

    def load_json(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_context(self, topic_id: int):
        niche = self.load_json(NICHE_FILE)
        topics = self.load_json(TOPICS_FILE)
        news = self.load_json(NEWS_FILE)

        selected_topic = next((t for t in topics if t["id"] == topic_id), None)
        if not selected_topic:
            raise ValueError(f"Topic with id {topic_id} not found in {TOPICS_FILE}")

        related_news = [
            n for n in news
            if any(word.lower() in ((n.get("title") or "") + " " + (n.get("description") or "")).lower()
                   for word in selected_topic["title"].split())
        ]

        query_text = selected_topic["title"]
        pdf_docs = self.vectordb.similarity_search(query_text, k=10)
        pdf_context = "\n".join([doc.page_content for doc in pdf_docs])

        return niche, selected_topic, related_news, pdf_context

    def clean_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.S)
            return json.loads(match.group()) if match else {}

    def generate_linkedin_content(self, topic, related_news, niche, audience, tone, pdf_context):
        pain_points_str = ', '.join([item['challenge'] for item in niche.get('customer_pain_points', [])])
        needs_str = ', '.join([item['need'] for item in niche.get('customer_needs', [])])

        prompt = f"""
        Create a LinkedIn post about the topic: "{topic['title']}".

        Context:
        - Industry: {niche.get("industry")}
        - Pain Points: {pain_points_str}
        - Needs: {needs_str}
        - Target Audience: {audience}
        - Desired Tone: {tone}
        - Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}
        - Reference Material (ICP/Niche PDF): {pdf_context}

        Requirements:
        - Write a professional caption (max 200 words).
        - Add 5-7 relevant hashtags.
        - Make it engaging and insight-driven.

        Output in JSON:
        {{
          "linkedin": {{
            "caption": "...",
            "hashtags": ["#", "#", "#"]
          }}
        }}
        """
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("linkedin", {})

    def generate_twitter_content(self, topic, related_news, niche, audience, tone, pdf_context):
        pain_points_str = ', '.join([item['challenge'] for item in niche.get('customer_pain_points', [])])
        needs_str = ', '.join([item['need'] for item in niche.get('customer_needs', [])])

        prompt = f"""
        Create a Twitter (X) post about the topic: "{topic['title']}".

        Context:
        - Industry: {niche.get("industry")}
        - Pain Points: {pain_points_str}
        - Needs: {needs_str}
        - Target Audience: {audience}
        - Desired Tone: {tone}
        - Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}
        - Reference Material (ICP/Niche PDF): {pdf_context}

        Requirements:
        - Keep under 280 characters.
        - Punchy, concise, and engaging.
        - Add 2-3 trending hashtags.

        Output in JSON:
        {{
          "twitter": {{
            "tweet": "...",
            "hashtags": ["#", "#"]
          }}
        }}
        """
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("twitter", {})

    def generate_youtube_content(self, topic, related_news, niche, audience, tone, pdf_context):
        pain_points_str = ', '.join([item['challenge'] for item in niche.get('customer_pain_points', [])])
        needs_str = ', '.join([item['need'] for item in niche.get('customer_needs', [])])

        prompt = f"""
        Create a YouTube video script intro and description for the topic: "{topic['title']}".

        Context:
        - Industry: {niche.get("industry")}
        - Pain Points: {pain_points_str}
        - Needs: {needs_str}
        - Target Audience: {audience}
        - Desired Tone: {tone}
        - Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}
        - Reference Material (ICP/Niche PDF): {pdf_context}

        Requirements:
        - Script intro (30-45 seconds).
        - Description (2-3 sentences).
        - Add 5-7 SEO-friendly tags.

        Output in JSON:
        {{
          "youtube": {{
            "script_intro": "...",
            "description": "...",
            "tags": ["tag1", "tag2", "tag3"]
          }}
        }}
        """
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("youtube", {})

    def run(self, topic_id: int, audience: str, tone: str):
        niche, topic, related_news, pdf_context = self.get_context(topic_id)

        linkedin_post = self.generate_linkedin_content(topic, related_news, niche, audience, tone, pdf_context)
        twitter_post = self.generate_twitter_content(topic, related_news, niche, audience, tone, pdf_context)
        youtube_post = self.generate_youtube_content(topic, related_news, niche, audience, tone, pdf_context)

        result = {
            "topic": topic,
            "linkedin": linkedin_post,
            "twitter": twitter_post,
            "youtube": youtube_post
        }

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print("✅ Content generated and saved to", OUTPUT_FILE)
        return result


def main():
    generator = ContentGenerator()

    topics = generator.load_json(TOPICS_FILE)
    if not topics:
        print("⚠️ No topics found. Run topic_generator first.")
        return

    print("\nAvailable Topics:")
    for t in topics:
        print(f"- ID {t['id']}: {t['title']}")

    topic_id = int(input("\nEnter the Topic ID: ").strip())
    tone = input("Specify the tone (e.g., professional, casual, bold): ").strip()
    audience = input("Preferred target audience (e.g., CXOs, Founders, Marketers): ").strip()

    generator.run(topic_id, audience, tone)


if __name__ == "__main__":
    main()
