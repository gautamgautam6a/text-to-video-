import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

NEWS_FILE = "./news/filtered_news.json"
TOPICS_FILE = "./topics/topics.json"


class TopicGenerator:
    def __init__(self, model="models/gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def load_news(self):
        if not os.path.exists(NEWS_FILE):
            raise FileNotFoundError(f"{NEWS_FILE} not found. Run trend_fetcher first.")
        with open(NEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_used_topics(self):
        if os.path.exists(TOPICS_FILE):
            with open(TOPICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_topics(self, all_topics):
        os.makedirs(os.path.dirname(TOPICS_FILE), exist_ok=True)
        with open(TOPICS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_topics, f, indent=4, ensure_ascii=False)

    def build_prompt(self, news_list, used_topics):
        news_text = json.dumps(news_list, indent=2, ensure_ascii=False)
        used_text = json.dumps([t["title"] for t in used_topics], indent=2, ensure_ascii=False)

        prompt = f"""
You are an AI assistant that creates social media content topics.

I will give you:
1. A list of recent news/trends.
2. A list of previously used topics.

Your job:
1. Analyze the news items and detect patterns, overlaps, or recurring themes.
2. Merge related items into a single concise topic if possible.
3. Generate exactly 3 NEW distinct, high-relevance topic options that are NOT in the previously used list.
4. Each topic should be short (max 12 words), actionable, and aligned with CXO-level strategy or industry opportunities.
5. Do not repeat the same wording across topics.

News Items:
{news_text}

Previously Used Topics:
{used_text}

Output ONLY in valid JSON as:
{{
  "topics": [
    {{"title": "Topic Title 1"}},
    {{"title": "Topic Title 2"}},
    {{"title": "Topic Title 3"}}
  ]
}}
"""
        return prompt

    def parse_response(self, response, next_id, news_list):
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.S)
            if not match:
                raise ValueError("No valid JSON found in response")
            data = json.loads(match.group())

        topics = []
        for i, t in enumerate(data.get("topics", []), start=0):
            title = t["title"].strip()

            # ✅ Normalize None -> ""
            related_news = []
            for n in news_list:
                text = f"{n.get('title') or ''} {n.get('description') or ''}".lower()
                if any(word.lower() in text for word in title.split()):
                    related_news.append(n)
            related_news = related_news[:5]  # cap at 5

            topics.append({
                "id": next_id + i,
                "title": title,
                "related_news": related_news
            })

        return topics


    def run(self):
        news_list = self.load_news()
        used_topics = self.load_used_topics()

        prompt = self.build_prompt(news_list, used_topics)
        response = self.llm.invoke(prompt).content.strip()

        next_id = max([t["id"] for t in used_topics], default=0) + 1
        new_topics = self.parse_response(response, next_id, news_list)

        all_topics = used_topics + new_topics
        self.save_topics(all_topics)

        print("✅ Generated new topics:")
        for t in new_topics:
            print(f"- {t['title']} ({len(t['related_news'])} related articles)")

        return new_topics


if __name__ == "__main__":
    generator = TopicGenerator()
    generator.run()
