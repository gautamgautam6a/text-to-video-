import re
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from TTS.api import TTS

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.7, google_api_key=google_api_key
)

# Prompts
researcher_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        f"You are a researcher.\n\
        Find the latest, most credible information about {{topic}} from reliable and up-to-date sources.\n\
        Return 10 concise, insight-driven bullet points that reflect the current state of the topic — with emphasis on:\n\
        - Hidden costs buyers or teams are silently tolerating\n\
        - Risks of inaction or delay (emotional, business, strategic)\n\
        - Blind spots the average decision-maker might miss\n\
        - Emerging trends or shifts in behavior, tech, or market expectations\n\
        - Unspoken challenges that could derail progress\n\
        \n\
        🧠 Each bullet should read like a wake-up call, not a Wikipedia fact.\n\
        🎯 Focus on framing problems in a way that creates urgency, discomfort, or FOMO.\n\
        ❌ Avoid fluff, generalities, or jargon — every bullet must deliver clarity or provoke reflection.\n\
        \n\
        At the end, include a 1-sentence insight summary that ties the key takeaway to buyer-level consequences."
    ),
)

script_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template=(
        f"Based on this research:\n{{research}}\n\nWrite a 30-second faceless video script about {{topic}}.\
        Use the following formatting structure and creative rules:\
        Scene formatting:\
        (Scene: [Brief but vivid description of visual footage — eg. 'a dimly lit office', 'fast-cut startup b-roll', 'typing hands on keyboard', etc.])\
        (0–5 seconds): [Narration text begins here — should immediately hook attention by highlighting one of the key risks or hidden costs from the research.]\
        (6–30 seconds): Continue with timestamped narration, roughly breaking the script into 10–20 second chunks. Weave in insights from the research to build tension and urgency.\
        Narration tone + pacing:\
        • Conversational but cinematic — like a sharp documentary voiceover.\
        • No jargon. No clichés. Speak like a smart peer who's seen things.\
        • Short, punchy lines that land hard.\
        • Vary emotional tone: tension → clarity → momentum → payoff.\
        Creative Direction:\
        ✅ Start with a strong emotional or intellectual hook — use one of the wake-up call points from the research.\
        ✅ Use visual metaphors and sensory details that can be represented easily by stock or animated footage.\
        ✅ Stack consequences — incorporate the blind spots and unspoken challenges from the research.\
        ✅ Don't just describe — reframe the problem using the emerging trends and insights provided.\
        ✅ End with a clear insight from the research summary, turned into a memorable one-liner or CTA.\
        Call to Action (CTA):\
        The ending CTA should feel like a realization based on the key research insight.\
        Video Length Guidance:\
        • Target: 30 seconds\
        • Use line breaks and timestamps to control pacing and visual flow"
    ),
)

# Step 1: Researcher
def researcher(topic: str, out_file="research.md"):
    research = llm.invoke(researcher_prompt.format(topic=topic))
    content = getattr(research, "content", str(research))
    if not content.strip():
        raise ValueError("Research generation failed: Empty output")
    Path(out_file).write_text(content, encoding="utf-8")
    return content  # return content for script use

# Step 2: Script generator (uses research)
def generate_script(topic: str, research: str, out_file="script.md"):
    script = llm.invoke(script_prompt.format(topic=topic, research=research))
    content = getattr(script, "content", str(script))
    if not content.strip():
        raise ValueError("Script generation failed: Empty output")
    Path(out_file).write_text(content, encoding="utf-8")
    return out_file

# Step 3: Voiceover generator (splits by timestamps)
def generate_voiceover(script_file="script.md", output_dir="voiceover",
                       voice_model="tts_models/en/ljspeech/tacotron2-DDC"):
    text = Path(script_file).read_text(encoding="utf-8")
    # Split narration into segments (ignores bold scene/timestamps)
    segments = re.findall(r"\*\*.*?\*\*|(.+?)(?=(\*\*|$))", text, re.DOTALL)
    segments = [seg[0].strip() for seg in segments if seg[0].strip()]

    if not segments:
        raise ValueError("No narration text found in the script.")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tts = TTS(voice_model, progress_bar=False)
    files = []
    for idx, segment in enumerate(segments, start=1):
        out_file = output_dir / f"part_{idx}.mp3"
        tts.tts_to_file(text=segment, file_path=str(out_file))
        files.append(str(out_file))

    return files

# Full pipeline
def generate_video(topic: str):
    research_text = researcher(topic)
    script_file = generate_script(topic, research_text)
    mp3_files = generate_voiceover(script_file)
    return {"research": "research.md", "script": script_file, "voiceover_files": mp3_files}

if __name__ == "__main__":
    result = generate_video("How to Build a Go-To-Market Strategy")
    print("Research File:", result["research"])
    print("Generated Script:", result["script"])
    print("Voiceover Files:", result["voiceover_files"])
