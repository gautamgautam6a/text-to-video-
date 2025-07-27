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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

def validate_setup():
    """Validate that all required dependencies and API keys are available"""
    logger.info("🔍 Validating setup...")
    required_keys = ["GEMINI_API_KEY", "ELEVENLABS_API_KEY", "PEXELS_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    
    for k in required_keys:
        if os.getenv(k):
            logger.info(f"✅ {k} found")
        else:
            logger.error(f"❌ {k} missing")
    
    if missing:
        logger.error(f"❌ Missing API keys: {', '.join(missing)}")
        return False
    
    try:
        import moviepy
        logger.info("✅ MoviePy available")
        import whisper
        logger.info("✅ Whisper available")
        return True
    except ImportError as e:
        logger.error(f"⚠ Missing dependency: {e}")
        return False

# Initialize APIs
google_api_key = os.getenv("GEMINI_API_KEY")
eleven_api = os.getenv("ELEVENLABS_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")

if not all([google_api_key, eleven_api, pexels_api_key]):
    if not validate_setup():
        exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Updated to latest model
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

def get_voice_id(name: str) -> str:
    """Get voice ID for ElevenLabs API"""
    return VOICE_IDS.get(name.lower(), VOICE_IDS["rachel"])

# =============================================================================
# CONTENT GENERATION
# =============================================================================

researcher_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Research the topic: {topic}\n"
        "Provide 10 key research insights in bullet points with specific examples.\n"
        "Focus on actionable, practical information that would be valuable for a short video.\n"
        "End with a 1-sentence summary that captures the core message."
    ),
)

script_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template=(
        "Based on this research:\n{research}\n\n"
        "Write an engaging 30-second video script for: {topic}\n"
        "The script should be conversational, punchy, and valuable to viewers.\n"
        "Output must be exactly in this format:\n"
        "(0-5 seconds): Hook - Start with an attention-grabbing statement or question\n"
        "(6-15 seconds): Problem/Context - Explain the key challenge or opportunity\n"
        "(16-25 seconds): Solution/Value - Provide the main insight or solution\n"
        "(26-30 seconds): Call-to-action - End with engagement or next steps\n"
        "Use short, punchy sentences. No extra labels, only narration text."
    ),
)

def research_topic(topic: str, out_file: str) -> str:
    """Research a topic using AI and save results"""
    logger.info(f"🔍 Researching: {topic}")
    try:
        response = llm.invoke(researcher_prompt.format(topic=topic))
        content = getattr(response, "content", str(response))
        Path(out_file).write_text(content, encoding="utf-8")
        logger.info(f"✅ Research saved to: {out_file}")
        return content
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        raise

def generate_script(topic: str, research: str, out_file: str) -> str:
    """Generate a timestamped script"""
    logger.info(f"📝 Generating script for: {topic}")
    try:
        response = llm.invoke(script_prompt.format(topic=topic, research=research))
        content = getattr(response, "content", str(response))
        Path(out_file).write_text(content, encoding="utf-8")
        logger.info(f"✅ Script saved to: {out_file}")
        return out_file
    except Exception as e:
        logger.error(f"❌ Script generation failed: {e}")
        raise

def ensure_timestamps(script_file: str) -> None:
    """Ensure script has proper timestamp format"""
    content = Path(script_file).read_text(encoding="utf-8").strip()
    
    # Check if already has timestamps
    if re.search(r"\(\d+[-–]\d+ seconds\):", content):
        logger.info("✅ Script already has timestamps")
        return
    
    logger.info("⏱ Adding timestamps to script")
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    
    if len(sentences) < 4:
        # If too few sentences, split differently
        words = content.split()
        chunk_size = len(words) // 4
        parts = [
            ' '.join(words[:chunk_size]),
            ' '.join(words[chunk_size:chunk_size*2]),
            ' '.join(words[chunk_size*2:chunk_size*3]),
            ' '.join(words[chunk_size*3:])
        ]
    else:
        parts = [
            '. '.join(sentences[:len(sentences)//4]),
            '. '.join(sentences[len(sentences)//4:len(sentences)//2]),
            '. '.join(sentences[len(sentences)//2:3*len(sentences)//4]),
            '. '.join(sentences[3*len(sentences)//4:])
        ]
    
    times = ["(0-5 seconds): ", "(6-15 seconds): ", "(16-25 seconds): ", "(26-30 seconds): "]
    final_script = "\n".join(f"{t}{p}." for t, p in zip(times, parts) if p.strip())
    
    Path(script_file).write_text(final_script, encoding="utf-8")
    logger.info(f"⏱ Timestamps added to {script_file}")

# =============================================================================
# AUDIO GENERATION
# =============================================================================

def generate_voiceover(script_file: str, output_dir: str, voice_name: str = "rachel") -> str:
    """Generate voiceover using ElevenLabs"""
    logger.info(f"🎙 Generating voiceover using {voice_name}")
    
    try:
        text = Path(script_file).read_text(encoding="utf-8")
        pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            raise ValueError("No timestamped narration found in script")
        
        # Combine all narration text
        narration = ' '.join([' '.join(seg[2].strip().splitlines()) for seg in matches])
        logger.info(f"Narration text: {narration[:100]}...")
        
        # Generate audio
        audio = generate(
            text=narration, 
            voice=get_voice_id(voice_name), 
            model="eleven_monolingual_v1"
        )
        
        # Save audio
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        audio_file = out_dir / "voiceover.mp3"
        
        with open(audio_file, "wb") as f:
            f.write(audio)
        
        logger.info(f"✅ Audio saved: {audio_file}")
        return str(audio_file)
        
    except Exception as e:
        logger.error(f"❌ Voiceover generation failed: {e}")
        raise

# =============================================================================
# SUBTITLE GENERATION
# =============================================================================

def extract_audio_for_whisper(audio_path: str, output_dir: str) -> str:
    """Convert audio to WAV format for Whisper processing"""
    try:
        from moviepy import AudioFileClip
        
        audio = AudioFileClip(audio_path)
        wav_path = Path(output_dir) / "audio_for_whisper.wav"
        audio.write_audiofile(str(wav_path), logger=None)
        audio.close()
        logger.info(f"✅ Audio converted for Whisper: {wav_path}")
        return str(wav_path)
    except Exception as e:
        logger.error(f"❌ Audio conversion failed: {e}")
        return audio_path

def generate_subtitles_from_audio(audio_path: str, output_dir: str) -> Optional[str]:
    """Generate subtitle file using Whisper"""
    try:
        logger.info("🎤 Loading Whisper model...")
        model = whisper.load_model("base")
        
        # Convert audio to WAV if needed
        wav_path = extract_audio_for_whisper(audio_path, output_dir)
        
        logger.info("🔍 Transcribing audio...")
        
        if not Path(wav_path).exists():
            logger.error(f"❌ Audio file not found: {wav_path}")
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
                
                # Format time for SRT
                start_srt = format_time_for_srt(start_time)
                end_srt = format_time_for_srt(end_time)
                
                f.write(f"{subtitle_index}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
                
                subtitle_index += 1
        
        logger.info(f"✅ Subtitles generated: {srt_path}")
        return str(srt_path)
        
    except Exception as e:
        logger.error(f"❌ Subtitle generation failed: {e}")
        return None

def format_time_for_srt(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def generate_simple_subtitles(script_file: str, output_dir: str) -> Optional[str]:
    """Generate subtitles directly from the script with estimated timing"""
    try:
        script_content = Path(script_file).read_text(encoding="utf-8")
        
        # Extract timestamped segments
        pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
        matches = re.findall(pattern, script_content, re.DOTALL)
        
        if not matches:
            logger.error("❌ No timestamped segments found in script")
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
        
        logger.info(f"✅ Simple subtitles generated: {srt_path}")
        return str(srt_path)
        
    except Exception as e:
        logger.error(f"❌ Simple subtitle generation failed: {e}")
        return None

# =============================================================================
# VIDEO PROCESSING & SEGMENT MANAGEMENT
# =============================================================================

def extract_keywords_from_segment(segment_text: str, main_topic: str) -> str:
    """Extract relevant keywords from script segment for video search"""
    # Remove common words and focus on key terms
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    words = re.findall(r'\b\w+\b', segment_text.lower())
    keywords = [w for w in words if w not in common_words and len(w) > 2]
    
    # Take top 3 keywords + main topic
    search_terms = keywords[:3] + [main_topic.lower()]
    return ' '.join(search_terms)

def generate_segment_search_queries(script_file: str, main_topic: str) -> List[Dict]:
    """Generate search queries for each script segment"""
    try:
        script_content = Path(script_file).read_text(encoding="utf-8")
        
        # Extract timestamped segments
        pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
        matches = re.findall(pattern, script_content, re.DOTALL)
        
        if not matches:
            logger.error("❌ No timestamped segments found in script")
            return []
        
        segment_queries = []
        for i, (start_sec, end_sec, text) in enumerate(matches):
            # Clean up text
            text = ' '.join(text.strip().splitlines())
            
            # Generate search query based on segment content
            if i == 0:  # Hook/Opening
                search_query = f"{main_topic} introduction professional"
            elif i == 1:  # Problem/Context
                keywords = extract_keywords_from_segment(text, main_topic)
                search_query = f"{keywords} business problem"
            elif i == 2:  # Solution/Value
                keywords = extract_keywords_from_segment(text, main_topic)
                search_query = f"{keywords} solution success"
            else:  # Call-to-action
                search_query = f"{main_topic} team collaboration success"
            
            segment_queries.append({
                'start': float(start_sec),
                'end': float(end_sec),
                'duration': float(end_sec) - float(start_sec),
                'text': text,
                'search_query': search_query,
                'segment_index': i
            })
        
        logger.info(f"✅ Generated {len(segment_queries)} segment search queries")
        return segment_queries
        
    except Exception as e:
        logger.error(f"❌ Segment query generation failed: {e}")
        return []

def download_video(url: str, path: str) -> bool:
    """Download video from URL"""
    try:
        logger.info(f"⬇️ Downloading video from: {url}")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        
        logger.info(f"✅ Video downloaded: {path}")
        return True
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def search_pexels_videos(query: str, per_page: int = 5) -> List[Dict]:
    """Search for videos on Pexels"""
    try:
        logger.info(f"🔍 Searching Pexels for: {query}")
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": pexels_api_key},
            params={
                "query": query, 
                "per_page": per_page, 
                "orientation": "landscape",
                "size": "medium"  # Optimize for faster downloads
            },
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        videos = data.get("videos", [])
        logger.info(f"✅ Found {len(videos)} videos for query: {query}")
        return videos
    except Exception as e:
        logger.error(f"❌ Pexels search failed for '{query}': {e}")
        return []

def get_video_download_url(video: Dict, quality: str = "hd") -> Optional[str]:
    """Extract download URL from Pexels video data"""
    for f in video.get("video_files", []):
        if f.get("quality") == quality:
            return f.get("link")
    
    # Fallback to first available
    video_files = video.get("video_files", [])
    return video_files[0].get("link") if video_files else None

def download_segment_videos(segment_queries: List[Dict], output_dir: str) -> List[Dict]:
    """Download videos for each script segment"""
    video_segments = []
    fallback_videos = []
    
    # First, try to get fallback videos
    fallback_searches = ["business meeting", "office work", "professional team", "corporate"]
    for fallback_query in fallback_searches:
        videos = search_pexels_videos(fallback_query, per_page=3)
        fallback_videos.extend(videos)
    
    output_path = Path(output_dir)
    
    for segment in segment_queries:
        logger.info(f"🎬 Processing segment {segment['segment_index']}: {segment['search_query']}")
        
        # Search for videos for this segment
        videos = search_pexels_videos(segment['search_query'], per_page=3)
        
        # If no videos found, use fallback
        if not videos and fallback_videos:
            videos = [fallback_videos[segment['segment_index'] % len(fallback_videos)]]
            logger.info(f"⚠ Using fallback video for segment {segment['segment_index']}")
        
        if videos:
            video = videos[0]  # Take the first/best result
            url = get_video_download_url(video)
            
            if url:
                video_filename = f"segment_{segment['segment_index']}_video.mp4"
                video_path = output_path / video_filename
                
                if download_video(url, str(video_path)):
                    segment['video_path'] = str(video_path)
                    segment['video_info'] = video
                    video_segments.append(segment)
                else:
                    logger.error(f"❌ Failed to download video for segment {segment['segment_index']}")
        else:
            logger.error(f"❌ No videos found for segment {segment['segment_index']}")
    
    logger.info(f"✅ Downloaded {len(video_segments)} segment videos")
    return video_segments

def create_segment_video_clips(video_segments: List[Dict]) -> List:
    """Create video clips for each segment with proper timing"""
    from moviepy import VideoFileClip, concatenate_videoclips
    
    segment_clips = []
    
    for segment in video_segments:
        try:
            logger.info(f"📹 Processing video for segment {segment['segment_index']}")
            
            # Load video
            video_clip = VideoFileClip(segment['video_path'])
            
            # Calculate the clip duration needed
            segment_duration = segment['duration']
            
            # If video is shorter than needed, loop it
            if video_clip.duration < segment_duration:
                loops_needed = int(segment_duration / video_clip.duration) + 1
                logger.info(f"🔄 Looping segment {segment['segment_index']} video {loops_needed} times")
                
                looped_clips = [video_clip] * loops_needed
                extended_clip = concatenate_videoclips(looped_clips)
                video_clip.close()  # Clean up original
                video_clip = extended_clip
            
            # Trim to exact duration needed using new API
            final_clip = video_clip.subclipped(0, segment_duration)
            
            # Set start time for this segment using new API
            final_clip = final_clip.with_start(segment['start'])
            
            segment_clips.append(final_clip)
            logger.info(f"✅ Segment {segment['segment_index']} video prepared: {segment_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to process video for segment {segment['segment_index']}: {e}")
            continue
    
    return segment_clips

def create_final_video_with_segments(
    video_segments: List[Dict], 
    audio_path: str, 
    subtitle_path: Optional[str], 
    output_path: str
) -> bool:
    """Create final video with multiple segment videos, audio and subtitles"""
    
    audio_clip = None
    final_video = None
    segment_clips = []
    
    try:
        from moviepy import AudioFileClip, CompositeVideoClip, concatenate_videoclips
        
        logger.info(f"🎵 Loading audio: {audio_path}")
        audio_clip = AudioFileClip(audio_path)
        
        # Create video clips for each segment
        segment_clips = create_segment_video_clips(video_segments)
        
        if not segment_clips:
            logger.error("❌ No video segments could be processed")
            return False
        
        logger.info(f"🎬 Concatenating {len(segment_clips)} video segments")
        
        # Concatenate all segment videos
        main_video = concatenate_videoclips(segment_clips, method="compose")
        
        # Ensure video matches audio duration
        if main_video.duration != audio_clip.duration:
            logger.info(f"⏱ Adjusting video duration from {main_video.duration:.2f}s to {audio_clip.duration:.2f}s")
            if main_video.duration > audio_clip.duration:
                main_video = main_video.subclipped(0, audio_clip.duration)
            else:
                # If somehow still shorter, extend the last segment
                last_segment_duration = audio_clip.duration - main_video.duration
                if last_segment_duration > 0:
                    logger.info(f"⏱ Extending last segment by {last_segment_duration:.2f}s")
                    # This is a fallback - ideally segments should sum to audio duration
        
        # Set audio to video
        video_with_audio = main_video.with_audio(audio_clip)
        
        # Add subtitles if available
        if subtitle_path and Path(subtitle_path).exists():
            logger.info("📝 Adding subtitles...")
            subtitle_clips = create_subtitle_clips_from_srt(subtitle_path, main_video.size)
            
            if subtitle_clips:
                final_video = CompositeVideoClip([video_with_audio] + subtitle_clips)
                logger.info(f"✅ Added {len(subtitle_clips)} subtitle clips")
            else:
                final_video = video_with_audio
        else:
            final_video = video_with_audio
        
        logger.info(f"🎥 Writing final video to: {output_path}")
        final_video.write_videofile(
            output_path, 
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            #verbose=False,
            logger=None
        )
        
        logger.info(f"✅ Final segmented video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Segmented video creation failed: {e}")
        return False
    finally:
        # Clean up resources
        if audio_clip:
            try:
                audio_clip.close()
            except:
                pass
        
        for clip in segment_clips:
            if clip:
                try:
                    clip.close()
                except:
                    pass
        
        if final_video:
            try:
                final_video.close()
            except:
                pass

def create_final_video_with_subtitles(
    video_path: str, 
    audio_path: str, 
    subtitle_path: Optional[str], 
    output_path: str
) -> bool:
    """Create final video with audio and subtitles (single video version)"""
    
    video_clip = None
    audio_clip = None
    final_video = None
    
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
        
        logger.info(f"🎬 Loading video: {video_path}")
        video_clip = VideoFileClip(video_path)
        
        logger.info(f"🎵 Loading audio: {audio_path}")
        audio_clip = AudioFileClip(audio_path)
        
        logger.info(f"Video duration: {video_clip.duration:.2f}s, Audio duration: {audio_clip.duration:.2f}s")

        # Match video duration to audio using new API
        if video_clip.duration > audio_clip.duration:
            video_clip = video_clip.subclipped(0, audio_clip.duration)
        elif video_clip.duration < audio_clip.duration:
            # Loop video to match audio duration
            loops_needed = int(audio_clip.duration / video_clip.duration) + 1
            logger.info(f"🔄 Looping video {loops_needed} times")
            
            looped_clips = [video_clip] * loops_needed
            extended_video = concatenate_videoclips(looped_clips)
            video_clip.close()  # Clean up original
            video_clip = extended_video.subclipped(0, audio_clip.duration)

        # Add subtitles if available
        if subtitle_path and Path(subtitle_path).exists():
            logger.info("📝 Adding subtitles...")
            subtitle_clips = create_subtitle_clips_from_srt(subtitle_path, video_clip.size)
            
            if subtitle_clips:
                # Create composite with video and subtitles
                final_video = CompositeVideoClip([video_clip] + subtitle_clips)
                # Set audio using direct attribute assignment (new API)
                final_video.audio = audio_clip
                logger.info(f"✅ Added {len(subtitle_clips)} subtitle clips")
            else:
                # No subtitles, just set audio on video
                video_clip.audio = audio_clip
                final_video = video_clip
        else:
            # No subtitles, just set audio on video
            video_clip.audio = audio_clip
            final_video = video_clip
        
        logger.info(f"🎥 Writing final video to: {output_path}")
        final_video.write_videofile(
            output_path, 
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        
        logger.info(f"✅ Final video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Video creation failed: {e}")
        return False
    finally:
        # Clean up resources
        for clip in [video_clip, audio_clip, final_video]:
            if clip:
                try:
                    clip.close()
                except:
                    pass

def create_subtitle_clips_from_srt(srt_path: str, video_size: tuple) -> List:
    """Create subtitle clips from SRT file"""
    try:
        from moviepy import TextClip
        
        # Parse SRT file
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        subtitle_clips = []
        blocks = content.split("\n\n")
        
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Parse timing
                time_line = lines[1]
                start_str, end_str = time_line.split(" --> ")
                
                start_time = parse_srt_time(start_str)
                end_time = parse_srt_time(end_str)
                duration = end_time - start_time
                
                # Get text
                text = " ".join(lines[2:])
                
                # Create text clip using new API
                txt_clip = TextClip(
                    text=text,
                    font_size=25,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    #font='Arial'
                ).with_duration(duration).with_start(start_time)
                
                # Position at bottom using new API
                txt_clip = txt_clip.with_position(('center', video_size[1] - 100))
                subtitle_clips.append(txt_clip)
        
        return subtitle_clips
        
    except Exception as e:
        logger.error(f"❌ Subtitle clip creation failed: {e}")
        return []

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    time_part, ms_part = time_str.split(",")
    h, m, s = map(int, time_part.split(":"))
    ms = int(ms_part)
    
    return h * 3600 + m * 60 + s + ms / 1000.0

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_complete_video_with_subtitles(
    topic: str, 
    voice_name: str = "rachel", 
    output_dir: str = "generated_video", 
    use_whisper: bool = True,
    use_multiple_videos: bool = True
) -> Optional[str]:
    """Complete video generation pipeline with subtitles and multiple segment videos"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting video generation for: {topic}")
        
        # Content generation
        research = research_topic(topic, str(output_path / "research.md"))
        script_file = generate_script(topic, research, str(output_path / "script.md"))
        ensure_timestamps(script_file)
        
        # Audio generation
        audio_file = generate_voiceover(script_file, str(output_path / "audio"), voice_name)
        
        # Subtitle generation
        subtitle_file = None
        if use_whisper:
            logger.info("Using Whisper for subtitle generation...")
            subtitle_file = generate_subtitles_from_audio(audio_file, str(output_path))
            if not subtitle_file:
                logger.warning("⚠ Whisper failed, falling back to script-based subtitles...")
                subtitle_file = generate_simple_subtitles(script_file, str(output_path))
        else:
            logger.info("Using script-based subtitle generation...")
            subtitle_file = generate_simple_subtitles(script_file, str(output_path))
        
        # Video acquisition and processing
        if use_multiple_videos:
            logger.info("🎬 Using multiple videos for different segments")
            
            # Generate segment-specific search queries
            segment_queries = generate_segment_search_queries(script_file, topic)
            if not segment_queries:
                logger.error("❌ Failed to generate segment queries")
                return None
            
            # Download videos for each segment
            video_segments = download_segment_videos(segment_queries, str(output_path))
            if not video_segments:
                logger.error("❌ Failed to download segment videos")
                return None
            
            # Create final video with multiple segments
            final_file = output_path / "final_video_multi_segments.mp4"
            if not create_final_video_with_segments(
                video_segments, audio_file, subtitle_file, str(final_file)
            ):
                return None
            
        else:
            logger.info("🎬 Using single video background")
            
            # Original single video approach
            videos = search_pexels_videos(topic)
            if not videos:
                logger.warning("No videos found for topic, trying fallback search...")
                videos = search_pexels_videos("business meeting")
            
            if not videos:
                logger.error("❌ No videos found")
                return None
                
            # Download video
            url = get_video_download_url(videos[0])
            if not url:
                logger.error("❌ No download URL found")
                return None
                
            video_file = output_path / "background.mp4"
            if not download_video(url, str(video_file)):
                return None
            
            # Create final video
            final_file = output_path / "final_video_single_background.mp4"
            if not create_final_video_with_subtitles(
                str(video_file), audio_file, subtitle_file, str(final_file)
            ):
                return None
            
        logger.info(f"🎉 Video generation complete: {final_file}")
        return str(final_file)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return None

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AI-powered videos with subtitles")
    parser.add_argument("topic", help="Video topic")
    parser.add_argument("--voice", default="rachel", choices=list(VOICE_IDS.keys()), 
                       help="Voice to use for narration")
    parser.add_argument("--output", default="generated_video", help="Output directory")
    parser.add_argument("--no-whisper", action="store_true", 
                       help="Use script-based subtitles instead of Whisper")
    parser.add_argument("--single-video", action="store_true",
                       help="Use single background video instead of multiple segment videos")
    
    args = parser.parse_args()
    
    if not validate_setup():
        exit(1)
    
    final_video = generate_complete_video_with_subtitles(
        topic=args.topic,
        voice_name=args.voice,
        output_dir=args.output,
        use_whisper=not args.no_whisper,
        use_multiple_videos=not args.single_video
    )
    
    if final_video:
        print(f"✅ Success! Video saved to: {final_video}")
    else:
        print("❌ Video generation failed")
        exit(1)

if __name__ == "__main__":
    main()