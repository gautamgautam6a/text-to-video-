import re
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from TTS.api import TTS

# Load .env
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Script Generation
script_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Write a 120-second faceless video script about {topic}. "
        "Format with timestamps and scene notes like this:\n"
        "**(Scene: ...)**\n**(0-5 seconds):** ... narration ..."
    ),
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=google_api_key)

def generate_script(topic: str, out_file="script.md"):
    script = llm.invoke(script_prompt.format(topic=topic)).content
    Path(out_file).write_text(script, encoding="utf-8")
    return out_file

# Voiceover (TTS)
def generate_voiceover(script_file="script.md", output_file="voiceover.mp3",
                       voice_model="tts_models/en/ljspeech/tacotron2-DDC"):
    text = Path(script_file).read_text(encoding="utf-8")
    cleaned = re.sub(r"\*\*.*?\*\*", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("No narration text found in the script.")
    tts = TTS(voice_model, progress_bar=False)
    tts.tts_to_file(text=cleaned, file_path=output_file)
    return output_file

def generate_video(topic: str):
    script_file = generate_script(topic)
    mp3_file = generate_voiceover(script_file)
    return {"script": script_file, "voiceover": mp3_file}

if __name__ == "__main__":
    result = generate_video("How to Build a Go-To-Market Strategy")
    print("Generated Script:", result["script"])
    print("Voiceover MP3:", result["voiceover"])
