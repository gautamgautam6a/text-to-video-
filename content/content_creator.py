import os
import google.generativeai as genai
from datetime import datetime
import json
import requests
from dotenv import load_dotenv

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def fetch_image_from_pexels(query):
    """Fetch first matching image URL from Pexels."""
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data["photos"]:
            return data["photos"][0]["src"]["large"]
    return None

def download_image(url, save_path):
    """Download image from a given URL."""
    img_data = requests.get(url).content
    with open(save_path, "wb") as f:
        f.write(img_data)

def create_posts_with_image(prompt_text):
    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Create folders if they don't exist
    os.makedirs("linkedin_posts", exist_ok=True)
    os.makedirs("twitter_posts", exist_ok=True)
    os.makedirs("linkedin_assets", exist_ok=True)

    # Step 1: Generate LinkedIn and Twitter post text
    social_prompt = f"""
    Create two separate social media posts based on the topic: "{prompt_text}".

    1. LinkedIn Post: Professional tone, 100-150 words, suitable for business audience.
    2. Twitter Post: Concise, under 280 characters, engaging.

    Respond strictly in JSON format without markdown:
    {{
      "linkedin": "LinkedIn post text here",
      "twitter": "Twitter post text here"
    }}
    """

    response = model.generate_content(social_prompt)

    cleaned_text = response.text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = "\n".join(
            line for line in cleaned_text.splitlines() if not line.strip().startswith("```")
        )

    try:
        posts_json = json.loads(cleaned_text)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON output from model: {cleaned_text}")

    linkedin_text = posts_json["linkedin"]
    twitter_text = posts_json["twitter"]

    # Save LinkedIn post
    linkedin_filename = f"linkedin_posts/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    with open(linkedin_filename, "w", encoding="utf-8") as f:
        f.write(linkedin_text)

    # Save Twitter post
    twitter_filename = f"twitter_posts/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    with open(twitter_filename, "w", encoding="utf-8") as f:
        f.write(twitter_text)

       # Step 2: Get LinkedIn image from Pexels
    image_url = fetch_image_from_pexels(prompt_text)
    image_filename = None
    if image_url:
        image_filename = f"linkedin_assets/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        download_image(image_url, image_filename)
        print(f"✅ LinkedIn image saved at: {image_filename}")
    else:
        print("⚠️ No image found for this topic.")

    print(f"✅ LinkedIn post saved at: {linkedin_filename}")
    print(f"✅ Twitter post saved at: {twitter_filename}")

    # Return values for use in main.py
    return linkedin_text, image_filename, twitter_text

