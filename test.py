import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from elevenlabs import generate, set_api_key

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

def validate_setup():
    print("🔍 Validating setup...")
    required_keys = ["GEMINI_API_KEY", "ELEVENLABS_API_KEY", "PEXELS_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    for k in required_keys:
        if os.getenv(k):
            print(f"✅ {k} found")
    if missing:
        print(f"❌ Missing API keys: {', '.join(missing)}")
        return False
    try:
        import moviepy
        print("✅ MoviePy available")
        return True
    except ImportError:
        print("⚠️ MoviePy not found")
        return False

google_api_key = os.getenv("GEMINI_API_KEY")
eleven_api = os.getenv("ELEVENLABS_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")

if not all([google_api_key, eleven_api, pexels_api_key]):
    if not validate_setup():
        exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=google_api_key
)
set_api_key(eleven_api)

VOICE_IDS = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "antoni": "ErXwobaYiN019PkySvjV",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ"
}

def get_voice_id(name: str):
    return VOICE_IDS.get(name.lower(), VOICE_IDS["rachel"])

# =============================================================================
# CONTENT GENERATION
# =============================================================================

researcher_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Research the topic: {topic}\nProvide 10 research insights in bullet points with examples.\n"
        "End with a 1-sentence summary."
    ),
)

script_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template=(
        "Based on this research:\n{research}\n\n"
        "Write a 30-second script for: {topic}\n"
        "Output must be exactly in this format:\n"
        "(0-5 seconds): ...\n(6-15 seconds): ...\n(16-25 seconds): ...\n(26-30 seconds): ...\n"
        "Short, punchy sentences, no extra labels, only narration."
    ),
)

def research_topic(topic: str, out_file: str) -> str:
    print(f"🔍 Researching: {topic}")
    response = llm.invoke(researcher_prompt.format(topic=topic))
    content = getattr(response, "content", str(response))
    Path(out_file).write_text(content, encoding="utf-8")
    print(f"✅ Research saved to: {out_file}")
    return content

def generate_script(topic: str, research: str, out_file: str) -> str:
    print(f"📝 Generating script for: {topic}")
    response = llm.invoke(script_prompt.format(topic=topic, research=research))
    content = getattr(response, "content", str(response))
    Path(out_file).write_text(content, encoding="utf-8")
    return out_file

def ensure_timestamps(script_file: str):
    content = Path(script_file).read_text(encoding="utf-8").strip()
    if re.search(r"\(\d+[-–]\d+ seconds\):", content):
        return
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    parts = ['. '.join(sentences[:len(sentences)//4]),
             '. '.join(sentences[len(sentences)//4:len(sentences)//2]),
             '. '.join(sentences[len(sentences)//2:3*len(sentences)//4]),
             '. '.join(sentences[3*len(sentences)//4:])]
    times = ["(0-5 seconds): ", "(6-15 seconds): ", "(16-25 seconds): ", "(26-30 seconds): "]
    final_script = "\n".join(f"{t}{p}." for t, p in zip(times, parts) if p.strip())
    Path(script_file).write_text(final_script, encoding="utf-8")
    print(f"⏱️ Timestamps ensured in {script_file}")

# =============================================================================
# AUDIO GENERATION
# =============================================================================

def generate_voiceover(script_file: str, output_dir: str, voice_name: str = "rachel") -> str:
    print(f"🎙️ Generating voiceover using {voice_name}")
    text = Path(script_file).read_text(encoding="utf-8")
    pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        raise ValueError("No timestamped narration found in script")
    narration = ' '.join([' '.join(seg[2].strip().splitlines()) for seg in matches])
    audio = generate(text=narration, voice=get_voice_id(voice_name), model="eleven_monolingual_v1")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    audio_file = out_dir / "voiceover.mp3"
    with open(audio_file, "wb") as f:
        f.write(audio)
    print(f"✅ Audio saved: {audio_file}")
    return str(audio_file)

# =============================================================================
# VIDEO CREATION (MoviePy v2 fixed)
# =============================================================================

def download_video(url: str, path: str) -> bool:
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"✅ Video downloaded: {path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def get_video_download_url(video: Dict, quality="hd") -> Optional[str]:
    for f in video.get("video_files", []):
        if f.get("quality") == quality:
            return f.get("link")
    return video.get("video_files", [{}])[0].get("link")

def search_pexels_videos(query: str, per_page=5) -> List[Dict]:
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": pexels_api_key},
            params={"query": query, "per_page": per_page, "orientation": "landscape"}
        )
        r.raise_for_status()
        data = r.json()
        return data.get("videos", [])
    except:
        return []


def create_final_video(video_path: str, audio_path: str, output_path: str) -> bool:
    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        
        print(f"🎬 Loading video: {video_path}")
        v = VideoFileClip(video_path)
        print(f"🎵 Loading audio: {audio_path}")
        a = AudioFileClip(audio_path)
        
        print(f"Video duration: {v.duration:.2f}s, Audio duration: {a.duration:.2f}s")

        # If video is shorter than audio, loop it
        if v.duration < a.duration:
            loops = int(a.duration / v.duration) + 1
            print(f"🔄 Looping video {loops} times to match audio duration")
            looped = concatenate_videoclips([v] * loops)
            # Use subclipped instead of subclip for modern MoviePy
            v = looped.subclipped(0, a.duration)
        else:
            # Trim video to match audio duration
            v = v.subclipped(0, a.duration)

        # Set audio to video using with_audio
        final_video = v.with_audio(a)
        
        print(f"🎥 Writing final video to: {output_path}")
        final_video.write_videofile(output_path)
        
        print(f"✅ Final video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_complete_video(topic: str, voice_name="rachel", output_dir="generated_video") -> Optional[str]:
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    try:
        research = research_topic(topic, str(output / "research.md"))
        script_file = generate_script(topic, research, str(output / "script.md"))
        ensure_timestamps(script_file)
        audio_file = generate_voiceover(script_file, str(output / "audio"), voice_name)
        videos = search_pexels_videos(topic)
        if not videos:
            videos = search_pexels_videos("business meeting")
        if not videos:
            return None
        url = get_video_download_url(videos[0])
        video_file = output / "background.mp4"
        if not download_video(url, str(video_file)):
            return None
        final_file = output / "final_video.mp4"
        if not create_final_video(str(video_file), audio_file, str(final_file)):
            return None
        print(f"🎉 Video ready at: {final_file}")
        return str(final_file)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

if __name__ == "__main__":
    if not validate_setup():
        exit(1)
    final = generate_complete_video("How to Build a Go-To-Market Strategy", "rachel")
    if not final:
        print("❌ Video generation failed")