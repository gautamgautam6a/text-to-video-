import os
import re
import json
import requests
import whisper
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
        import whisper
        print("✅ Whisper available")
        return True
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")
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
# SUBTITLE GENERATION
# =============================================================================

def extract_audio_for_whisper(audio_path: str, output_dir: str) -> str:
    """Convert audio to WAV format for Whisper processing"""
    try:
        from moviepy import AudioFileClip
        
        audio = AudioFileClip(audio_path)
        wav_path = Path(output_dir) / "audio_for_whisper.wav"
        # Fixed: Remove verbose parameter for newer MoviePy versions
        audio.write_audiofile(str(wav_path), logger=None)
        audio.close()
        print(f"✅ Audio converted for Whisper: {wav_path}")
        return str(wav_path)
    except Exception as e:
        print(f"❌ Audio conversion failed: {e}")
        # Return original path if conversion fails
        return audio_path

def generate_subtitles_from_audio(audio_path: str, output_dir: str) -> str:
    """Generate subtitle file using Whisper"""
    try:
        print("🎤 Loading Whisper model...")
        model = whisper.load_model("base")  # You can use "small", "medium", "large" for better accuracy
        
        # Convert audio to WAV if needed
        wav_path = extract_audio_for_whisper(audio_path, output_dir)
        
        print("🔍 Transcribing audio...")
        print(f"Using audio file: {wav_path}")
        
        # Check if file exists
        if not Path(wav_path).exists():
            print(f"❌ Audio file not found: {wav_path}")
            return None
            
        result = model.transcribe(wav_path, word_timestamps=True)
        
        # Generate SRT format subtitles
        srt_path = Path(output_dir) / "subtitles.srt"
        
        with open(srt_path, "w", encoding="utf-8") as f:
            subtitle_index = 1
            
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                
                # Format time for SRT (HH:MM:SS,mmm)
                start_srt = format_time_for_srt(start_time)
                end_srt = format_time_for_srt(end_time)
                
                f.write(f"{subtitle_index}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
                
                subtitle_index += 1
        
        print(f"✅ Subtitles generated: {srt_path}")
        return str(srt_path)
        
    except Exception as e:
        print(f"❌ Subtitle generation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def format_time_for_srt(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def parse_srt_file(srt_path: str) -> List[Dict]:
    """Parse SRT file into subtitle segments"""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        segments = []
        blocks = content.split("\n\n")
        
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Parse time range
                time_line = lines[1]
                start_str, end_str = time_line.split(" --> ")
                
                start_time = parse_srt_time(start_str)
                end_time = parse_srt_time(end_str)
                
                # Get text (may be multiple lines)
                text = " ".join(lines[2:])
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
        
        return segments
        
    except Exception as e:
        print(f"❌ SRT parsing failed: {e}")
        return []

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    time_part, ms_part = time_str.split(",")
    h, m, s = map(int, time_part.split(":"))
    ms = int(ms_part)
    
    return h * 3600 + m * 60 + s + ms / 1000.0

def create_subtitle_clips(subtitle_segments: List[Dict], video_size: tuple):
    """Create TextClip objects for each subtitle segment"""
    from moviepy import TextClip
    
    subtitle_clips = []
    
    for segment in subtitle_segments:
        # Calculate duration
        duration = segment["end"] - segment["start"]
        
        # Create text clip with professional styling using new MoviePy syntax
        txt_clip = TextClip(
            text=segment["text"],
            font_size=35,        # Large, readable font
            color='white',       # White text
            stroke_color='black', # Black outline
            stroke_width=2       # Thick outline for contrast
        ).with_duration(duration).with_start(segment["start"])
        
        # Position at bottom of screen with some padding
        txt_clip = txt_clip.with_position(('center', video_size[1] - 150))
        
        subtitle_clips.append(txt_clip)
    
    return subtitle_clips

# =============================================================================
# VIDEO CREATION
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

def create_final_video_with_subtitles(video_path: str, audio_path: str, subtitle_path: str, output_path: str) -> bool:
    """Create final video with subtitles"""
    v, a, final_video = None, None, None
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip
        
        print(f"🎬 Loading video: {video_path}")
        v = VideoFileClip(video_path)
        print(f"🎵 Loading audio: {audio_path}")
        a = AudioFileClip(audio_path)
        
        print(f"Video duration: {v.duration:.2f}s, Audio duration: {a.duration:.2f}s")

        # Simple approach: just trim or extend to match audio duration
        if v.duration > a.duration:
            # Video is longer, trim it
            print(f"✂️ Trimming video to match audio duration")
            v = v.subclipped(0, a.duration)
        elif v.duration < a.duration:
            # Video is shorter, loop it
            loops = int(a.duration / v.duration) + 1
            print(f"🔄 Creating {loops} loops to match audio duration")
            
            # Create individual clips and concatenate
            clips_to_concat = []
            remaining_duration = a.duration
            
            for i in range(loops):
                if remaining_duration <= 0:
                    break
                    
                if remaining_duration >= v.duration:
                    # Full clip
                    clips_to_concat.append(v)
                    remaining_duration -= v.duration
                else:
                    # Partial clip for the remainder
                    clips_to_concat.append(v.subclipped(0, remaining_duration))
                    remaining_duration = 0
            
            if len(clips_to_concat) > 1:
                from moviepy import concatenate_videoclips
                v = concatenate_videoclips(clips_to_concat)
            else:
                v = clips_to_concat[0]

        # Set audio to video using new syntax
        video_with_audio = v.with_audio(a)
        
        # Add subtitles if subtitle file exists
        if subtitle_path and Path(subtitle_path).exists():
            print("📝 Adding subtitles...")
            print(f"Subtitle file: {subtitle_path}")
            subtitle_segments = parse_srt_file(subtitle_path)
            print(f"Found {len(subtitle_segments)} subtitle segments")
            
            if subtitle_segments:
                subtitle_clips = create_subtitle_clips(subtitle_segments, v.size)
                print(f"Created {len(subtitle_clips)} subtitle clips")
                
                # Composite video with subtitles using new syntax
                final_video = CompositeVideoClip([video_with_audio] + subtitle_clips)
            else:
                print("⚠️ No subtitle segments found, using video without subtitles")
                final_video = video_with_audio
        else:
            print("⚠️ No subtitle file found, using video without subtitles")
            final_video = video_with_audio
        
        print(f"🎥 Writing final video with subtitles to: {output_path}")
        final_video.write_videofile(output_path)
        
        print(f"✅ Final video with subtitles created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Video creation with subtitles failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up resources
        for clip in [v, a, final_video]:
            if clip:
                try:
                    clip.close()
                except:
                    pass

def create_final_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """Original video creation function without subtitles"""
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
            v = looped.subclipped(0, a.duration)
        else:
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
    """Original pipeline function without subtitles"""
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

def generate_simple_subtitles(script_file: str, output_dir: str) -> str:
    """Generate subtitles directly from the script with estimated timing"""
    try:
        # Read the script
        script_content = Path(script_file).read_text(encoding="utf-8")
        
        # Extract timestamped segments
        pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
        matches = re.findall(pattern, script_content, re.DOTALL)
        
        if not matches:
            print("❌ No timestamped segments found in script")
            return None
            
        srt_path = Path(output_dir) / "subtitles.srt"
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (start_sec, end_sec, text) in enumerate(matches, 1):
                # Clean up text
                text = ' '.join(text.strip().splitlines())
                
                # Format times
                start_time = format_time_for_srt(float(start_sec))
                end_time = format_time_for_srt(float(end_sec))
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        print(f"✅ Simple subtitles generated: {srt_path}")
        return str(srt_path)
        
    except Exception as e:
        print(f"❌ Simple subtitle generation failed: {e}")
        return None

def generate_complete_video_with_subtitles(topic: str, voice_name="rachel", output_dir="generated_video", use_whisper=True) -> Optional[str]:
    """Enhanced pipeline function with subtitles"""
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    try:
        # Your existing pipeline steps
        research = research_topic(topic, str(output / "research.md"))
        script_file = generate_script(topic, research, str(output / "script.md"))
        ensure_timestamps(script_file)
        audio_file = generate_voiceover(script_file, str(output / "audio"), voice_name)
        
        # Generate subtitles
        print("🎬 Generating subtitles...")
        if use_whisper:
            print("Using Whisper for subtitle generation...")
            subtitle_file = generate_subtitles_from_audio(audio_file, str(output))
            if not subtitle_file:
                print("⚠️ Whisper failed, falling back to script-based subtitles...")
                subtitle_file = generate_simple_subtitles(script_file, str(output))
        else:
            print("Using script-based subtitle generation...")
            subtitle_file = generate_simple_subtitles(script_file, str(output))
        
        # Get video
        videos = search_pexels_videos(topic)
        if not videos:
            videos = search_pexels_videos("business meeting")
        if not videos:
            return None
            
        url = get_video_download_url(videos[0])
        video_file = output / "background.mp4"
        
        if not download_video(url, str(video_file)):
            return None
        
        # Create final video with subtitles
        final_file = output / "final_video_with_subtitles.mp4"
        if not create_final_video_with_subtitles(str(video_file), audio_file, subtitle_file, str(final_file)):
            return None
            
        print(f"🎉 Video with subtitles ready at: {final_file}")
        return str(final_file)
        
    except Exception as e:
        print(f"❌ Pipeline with subtitles failed: {e}")
        return None

if __name__ == "__main__":
    if not validate_setup():
        exit(1)
    
    # Now use Whisper subtitles (FFmpeg is installed)
    final = generate_complete_video_with_subtitles("How to Build a Go-To-Market Strategy", "rachel", use_whisper=True)
    
    # Fallback options:
    # final = generate_complete_video_with_subtitles("How to Build a Go-To-Market Strategy", "rachel", use_whisper=False)  # Script-based
    # final = generate_complete_video("How to Build a Go-To-Market Strategy", "rachel")  # No subtitles
    
    if not final:
        print("❌ Video generation failed")