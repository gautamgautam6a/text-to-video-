import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from prompts import LINKEDIN_PROMPT, TWITTER_PROMPT, RESEARCH_PROMPT

load_dotenv()

# ----------- GEMINI CONFIG -----------
# LLM for text
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Direct Gemini API for image
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ----------- UTILITIES -----------

def save_markdown(content: str, folder: str) -> str:
    """Save text content into a timestamped markdown file."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def generate_gemini_image(topic: str):
    """Generates a LinkedIn/Twitter style image from Gemini image model."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=f"High-quality, professional social-media-ready image about: {topic}",
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    image_path = None
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            image = Image.open(BytesIO(part.inline_data.data))
            os.makedirs("generated_images", exist_ok=True)

            # sanitize filename (remove ? and other illegal chars)
            safe_topic = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in topic)
            image_path = f"generated_images/{safe_topic}_{int(time.time())}.png"

            image.save(image_path)
            print(f"✅ Image saved at: {image_path}")

    if not image_path:
        print("⚠ No image data found in response.")

    return image_path

# ----------- AGENTS -----------

def research_agent():
    """Research trending topic for predefined business niche using Gemini, avoiding repeats."""
    topics_file = "topics.json"

    # Load existing topics
    if os.path.exists(topics_file):
        with open(topics_file, "r", encoding="utf-8") as f:
            try:
                topics_data = json.load(f)
            except json.JSONDecodeError:
                topics_data = []
    else:
        topics_data = []

    used_topics = [t["topic"] for t in topics_data]

    # Build dynamic prompt with exclusions
    today = datetime.now().strftime("%Y-%m-%d")
    exclusion_text = "\n".join(f"- {t}" for t in used_topics) if used_topics else "None yet"

    dynamic_prompt = RESEARCH_PROMPT + f"""

🚫 Do NOT return any of these already-used topics:
{exclusion_text}

📅 Today’s date: {today}
"""

    # Run Gemini
    prompt_template = PromptTemplate(input_variables=[], template=dynamic_prompt)
    response = llm_gemini.invoke(prompt_template.format())
    topic_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # Save if unique
    if topic_text not in used_topics:
        topics_data.append({
            "topic": topic_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        with open(topics_file, "w", encoding="utf-8") as f:
            json.dump(topics_data, f, indent=4)

    return topic_text



def linkedin_agent(topic: str):
    """Generate LinkedIn post + image for given topic."""
    prompt_template = PromptTemplate(input_variables=["topic"], template=LINKEDIN_PROMPT)

    linkedin_post = llm_gemini.invoke(prompt_template.format(topic=topic))
    linkedin_post_text = linkedin_post.content if hasattr(linkedin_post, "content") else str(linkedin_post)

    md_path = save_markdown(linkedin_post_text, "linkedin_posts")
    image_path = generate_gemini_image(topic)

    return linkedin_post_text, md_path, image_path


def twitter_agent(topic: str):
    """Generate Twitter post for given topic."""
    prompt_template = PromptTemplate(input_variables=["topic"], template=TWITTER_PROMPT)

    twitter_post = llm_gemini.invoke(prompt_template.format(topic=topic))
    twitter_post_text = twitter_post.content if hasattr(twitter_post, "content") else str(twitter_post)

    md_path = save_markdown(twitter_post_text, "twitter_posts")

    return twitter_post_text, md_path
