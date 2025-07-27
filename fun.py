import os
import requests
import whisper
from elevenlabs import generate, set_api_key
from moviepy import (
    concatenate_videoclips, VideoFileClip, CompositeVideoClip,
    TextClip, AudioFileClip
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import nltk
import time
import json

# Load environment variables
load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
set_api_key(ELEVENLABS_API_KEY)

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

# Initialize Gemini via LangChain
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# === Step 1: Segment the script ===
def segment_script(script):
    return sent_tokenize(script)

# === Step 2: Generate search queries for each segment ===
def generate_queries(script_segments):
    prompt = ChatPromptTemplate.from_template(
        "Generate a short search query for stock video footage for the following sentence: {segment}"
    )
    queries = []
    for segment in script_segments:
        chain = prompt | chat
        response = chain.invoke({"segment": segment})
        queries.append(response.content.strip())
    return queries

# === Step 3: Search for videos using Pexels API ===
def search_video(query):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page=1"
    response = requests.get(url, headers=headers)
    data = response.json()
    try:
        return data["videos"][0]["video_files"][0]["link"]
    except (IndexError, KeyError):
        return None

# === Step 4: Download and trim video ===
def download_and_trim_video(url, output_path, duration):
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)

    video = VideoFileClip(output_path)
    if video.duration < duration:
        # Loop the video if it's too short
        loop_count = int(duration // video.duration) + 1
        video = concatenate_videoclips([video] * loop_count)
    trimmed = video.subclipped(0, duration)
    trimmed.write_videofile(output_path.replace(".mp4", "_trimmed.mp4"), codec="libx264")
    return output_path.replace(".mp4", "_trimmed.mp4")

# === Step 5: Generate voiceover using ElevenLabs ===
def generate_voice(script, output_path, voice="Bella"):
    audio = generate(text=script, voice=voice, model="eleven_multilingual_v2")
    with open(output_path, "wb") as f:
        f.write(audio)

# === Step 6: Transcribe for subtitles ===
def generate_subtitles(audio_path, srt_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            srt_file.write(f"{text}\n\n")

def format_srt_time(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# === Step 7: Overlay subtitles ===
def overlay_subtitles(video, srt_path):
    from moviepy.video.tools.subtitles import SubtitlesClip
    def generator(txt):
        return TextClip(txt, font="Arial", fontsize=24, color="white")
    subs = SubtitlesClip(srt_path, generator)
    return CompositeVideoClip([video, subs.with_position(("center", "bottom"))])

# === Step 8: Assemble final video ===
def assemble_video(script):
    script_segments = segment_script(script)
    queries = generate_queries(script_segments)

    downloaded_clips = []
    segment_duration = 5  # seconds
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query}")
        url = search_video(query)
        if url:
            video_path = f"segment_{i}.mp4"
            trimmed_path = download_and_trim_video(url, video_path, segment_duration)
            clip = VideoFileClip(trimmed_path).subclipped(0, segment_duration)
            downloaded_clips.append(clip)
        else:
            print(f"No video found for: {query}")

    if not downloaded_clips:
        raise Exception("No videos downloaded")

    concatenated = concatenate_videoclips(downloaded_clips, method="compose")

    # Voice Generation
    audio_path = "output_audio.mp3"
    generate_voice(script, audio_path)

    # Subtitles
    srt_path = "output_subs.srt"
    generate_subtitles(audio_path, srt_path)

    # Sync durations
    audio_clip = AudioFileClip(audio_path).subclipped(0, concatenated.duration)
    final_video = concatenated.with_audio(audio_clip)
    final_video_with_subs = overlay_subtitles(final_video, srt_path)

    # Export
    final_video_with_subs.write_videofile(
        "final_video.mp4",
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )

# === Run the pipeline ===
if __name__ == "__main__":
    script_text = """
    Artificial Intelligence is transforming the way we interact with technology.
    From self-driving cars to smart assistants, AI is at the core of modern innovation.
    It enables machines to learn from data and improve over time.
    With AI, businesses can automate tasks, enhance customer experiences, and make better decisions.
    The future of AI holds endless possibilities, reshaping industries and our daily lives.
    """
    assemble_video(script_text)
