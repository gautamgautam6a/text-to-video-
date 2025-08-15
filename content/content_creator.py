import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import LINKEDIN_PROMPT, TWITTER_PROMPT

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Configure Gemini LLM
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ----------- TOOLS -----------

def save_markdown(content: str, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def get_pexels_image(query: str) -> str:
    """Fetches first Pexels image for given query and saves locally."""
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(
        f"https://api.pexels.com/v1/search?query={query}&per_page=1",
        headers=headers
    )
    data = response.json()
    if not data.get("photos"):
        raise ValueError("No images found on Pexels for the given query.")
    
    image_url = data["photos"][0]["src"]["large"]
    os.makedirs("linkedin_assets", exist_ok=True)
    filename = f"linkedin_assets/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    img_data = requests.get(image_url).content
    with open(filename, "wb") as handler:
        handler.write(img_data)
    return filename

# ----------- AGENTS -----------

def linkedin_agent(topic: str):
    prompt_template = PromptTemplate(input_variables=["topic"], template=LINKEDIN_PROMPT)

    tools = [
        Tool(
            name="Save LinkedIn Post",
            func=lambda text: save_markdown(text, "linkedin_posts"),
            description="Save the LinkedIn post to a markdown file"
        ),
        Tool(
            name="Get LinkedIn Image",
            func=lambda text: get_pexels_image(text),
            description="Search and download a relevant LinkedIn image from Pexels"
        )
    ]

    agent = initialize_agent(
        tools,
        llm_gemini,
        agent="zero-shot-react-description",
        verbose=True
    )

    linkedin_post = llm_gemini.predict(prompt_template.format(topic=topic))
    md_path = save_markdown(linkedin_post, "linkedin_posts")
    image_path = get_pexels_image(topic)

    return linkedin_post, md_path, image_path


def twitter_agent(topic: str):
    prompt_template = PromptTemplate(input_variables=["topic"], template=TWITTER_PROMPT)

    tools = [
        Tool(
            name="Save Twitter Post",
            func=lambda text: save_markdown(text, "twitter_posts"),
            description="Save the Twitter post to a markdown file"
        )
    ]

    agent = initialize_agent(
        tools,
        llm_gemini,
        agent="zero-shot-react-description",
        verbose=True
    )

    twitter_post = llm_gemini.predict(prompt_template.format(topic=topic))
    md_path = save_markdown(twitter_post, "twitter_posts")

    return twitter_post, md_path
