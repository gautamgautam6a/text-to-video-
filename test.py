import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from elevenlabs import generate, set_api_key

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
eleven_api = os.getenv("ELEVENLABS_API_KEY")

if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not eleven_api:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")

# Initialize clients
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.7, google_api_key=google_api_key
)
set_api_key(eleven_api)

# Common ElevenLabs voice IDs
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

def get_voice_id(voice_name: str) -> str:
    """Get voice ID from name, with fallback to Rachel"""
    return VOICE_IDS.get(voice_name.lower(), VOICE_IDS["rachel"])

# Prompts
researcher_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are an expert researcher specializing in business intelligence and market analysis.\n"
        "Research the topic: {topic}\n\n"
        "Provide 10 research-backed insights focusing on:\n"
        "• Hidden costs and overlooked expenses\n"
        "• Critical risks of delayed action\n"
        "• Common blind spots in decision-making\n"
        "• Emerging industry trends and disruptions\n"
        "• Unspoken operational challenges\n"
        "• Competitive advantages being missed\n"
        "• Resource allocation inefficiencies\n"
        "• Customer behavior shifts\n"
        "• Technology impact and automation threats\n"
        "• Market timing considerations\n\n"
        "Format each insight as an actionable bullet point with specific examples when possible.\n"
        "End with a compelling 1-sentence summary highlighting the most critical business consequence."
    ),
)

script_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template=(
        "Based on this research:\n{research}\n\n"
        "Create a compelling 30-second video script for: {topic}\n\n"
        "STRUCTURE (follow exactly):\n"
        "(0-5 seconds): Start with a shocking statistic, question, or contrarian statement\n"
        "(6-15 seconds): Build urgency with hidden costs, risks, or missed opportunities\n"
        "(16-25 seconds): Reveal the solution, trend, or strategic advantage\n"
        "(26-30 seconds): End with a memorable call-to-action or thought-provoking statement\n\n"
        "STYLE REQUIREMENTS:\n"
        "✓ Conversational yet authoritative tone\n"
        "✓ Short, punchy sentences (max 12 words each)\n"
        "✓ Specific numbers and data points when available\n"
        "✓ Emotional progression: surprise → concern → clarity → action\n"
        "✓ Avoid jargon, buzzwords, and generic statements\n"
        "✓ Include natural pauses with [...] for dramatic effect\n"
        "✓ Do NOT include section labels like 'HOOK', 'TENSION' etc.\n\n"
        "Output ONLY the timestamped narration text that will be spoken."
    ),
)

# Step 1: Research
def researcher(topic: str, out_file="research.md"):
    """Generate comprehensive research on the given topic."""
    print(f"🔍 Researching: {topic}")
    
    research = llm.invoke(researcher_prompt.format(topic=topic))
    content = getattr(research, "content", str(research))
    
    if not content.strip():
        raise ValueError("Research generation failed: Empty output")
    
    Path(out_file).write_text(content, encoding="utf-8")
    print(f"✅ Research saved to: {out_file}")
    return content

def clean_script_content(content: str) -> str:
    """Clean script content by removing section headers and keeping only narration."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Keep timestamp lines and narration, skip section headers
        if line.startswith('(') or (line and not line.isupper() and not line.endswith(':')):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Step 2: Script Generation
def generate_script(topic: str, research: str, out_file="script.md"):
    """Generate a video script based on research."""
    print(f"📝 Generating script for: {topic}")
    
    script = llm.invoke(script_prompt.format(topic=topic, research=research))
    content = getattr(script, "content", str(script))
    
    if not content.strip():
        raise ValueError("Script generation failed: Empty output")
    
    # Clean the content to remove section headers
    cleaned_content = clean_script_content(content)
    
    Path(out_file).write_text(cleaned_content, encoding="utf-8")
    print(f"✅ Script saved to: {out_file}")
    return out_file

# Step 3: Voice Generation
def generate_voiceover(script_file="script.md", output_dir="voiceover", voice_name="rachel"):
    """Generate a single continuous voiceover audio file from script."""
    print(f"🎙️ Generating voiceover with voice: {voice_name}")
    
    # Read script
    text = Path(script_file).read_text(encoding="utf-8")
    
    # Extract timestamped segments and combine into single narration
    pattern = r"\((\d+)[–-](\d+) seconds\):\s*(.+?)(?=\n\(|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        raise ValueError("No timestamped narration found in script.")
    
    # Clean matches to remove section headers and combine all narration
    narration_parts = []
    timestamp_map = []
    
    for start, end, content in matches:
        # Remove section headers like "HOOK", "TENSION", "INSIGHT", "PAYOFF"
        lines = content.strip().split('\n')
        narration_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and section headers
            if line and not line.isupper() and not line.endswith(':'):
                narration_lines.append(line)
        
        # Join the narration lines for this segment
        segment_narration = ' '.join(narration_lines)
        if segment_narration:  # Only add if there's actual narration
            narration_parts.append(segment_narration)
            timestamp_map.append({
                "start": int(start),
                "end": int(end),
                "duration": int(end) - int(start),
                "text": segment_narration,
                "word_count": len(segment_narration.split())
            })
    
    # Combine all narration with natural pauses
    full_narration = ' '.join(narration_parts)
    
    # Get voice ID
    voice_id = get_voice_id(voice_name)
    print(f"Using voice ID: {voice_id}")
    print(f"📝 Full script: {full_narration[:100]}...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        print("🔊 Generating single continuous audio file...")
        
        # Generate single audio file using the working approach
        audio = generate(
            text=full_narration,
            voice=voice_id,
            model="eleven_monolingual_v1"
        )
        
        # Save single audio file
        audio_file = output_dir / "full_voiceover.mp3"
        with open(audio_file, "wb") as f:
            f.write(audio)
        
        print(f"✅ Saved: {audio_file}")
        
        # Save timeline data for reference
        timeline_file = output_dir / "timeline.json"
        timeline_data = {
            "voice_name": voice_name,
            "voice_id": voice_id,
            "audio_file": str(audio_file),
            "full_text": full_narration,
            "total_words": len(full_narration.split()),
            "total_duration": 30,
            "segments_info": timestamp_map
        }
        
        with open(timeline_file, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)
        
        print(f"✅ Timeline saved to: {timeline_file}")
        print(f"🎉 Generated single audio file successfully!")
        
        return [str(audio_file)]  # Return as list to maintain compatibility
        
    except Exception as e:
        print(f"❌ Failed to generate audio: {e}")
        raise

# Simple function to generate audio from any text
def text_to_audio(text: str, output_file: str = "voiceover.mp3", voice_name: str = "rachel"):
    """Generate audio directly from text."""
    print(f"🎙️ Converting text to audio with voice: {voice_name}")
    
    voice_id = get_voice_id(voice_name)
    print(f"Using voice ID: {voice_id}")
    
    try:
        audio = generate(
            text=text,
            voice=voice_id,
            model="eleven_monolingual_v1"
        )
        
        with open(output_file, "wb") as f:
            f.write(audio)
        
        print(f"✅ Audio saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Failed to generate audio: {e}")
        raise

# Full pipeline
def generate_video(topic: str, voice_name="rachel", output_dir="output"):
    """Complete pipeline to generate video content."""
    print(f"\n🚀 Starting video generation for: '{topic}'")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Step 1: Research
        research_file = output_path / "research.md"
        research_text = researcher(topic, str(research_file))
        
        # Step 2: Script
        script_file = output_path / "script.md"
        generate_script(topic, research_text, str(script_file))
        
        # Step 3: Voiceover
        voiceover_dir = output_path / "voiceover"
        mp3_files = generate_voiceover(str(script_file), str(voiceover_dir), voice_name)
        
        # Summary
        result = {
            "topic": topic,
            "research": str(research_file),
            "script": str(script_file),
            "voiceover_files": mp3_files,
            "voiceover_directory": str(voiceover_dir),
            "timeline": str(voiceover_dir / "timeline.json"),
            "audio_file": mp3_files[0] if mp3_files else None  # Single file path
        }
        
        print("\n" + "=" * 60)
        print("🎉 VIDEO GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📁 Output Directory: {output_dir}")
        print(f"📄 Research File: {result['research']}")
        print(f"📝 Script File: {result['script']}")
        print(f"🎵 Audio File: {result['voiceover_files'][0]}")  # Single file now
        print(f"📊 Timeline Data: {result['timeline']}")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

# Test available voices
def test_voices():
    """Test different voices to see which ones work."""
    test_text = "Testing voice quality and clarity."
    
    print("🧪 Testing available voices...")
    working_voices = []
    
    for voice_name, voice_id in VOICE_IDS.items():
        try:
            print(f"Testing {voice_name} ({voice_id})...")
            audio = generate(
                text=test_text,
                voice=voice_id,
                model="eleven_monolingual_v1"
            )
            
            # Save test file
            test_file = f"test_{voice_name}.mp3"
            with open(test_file, "wb") as f:
                f.write(audio)
            
            working_voices.append(voice_name)
            print(f"✅ {voice_name} - Working! Saved as {test_file}")
            
        except Exception as e:
            print(f"❌ {voice_name} - Failed: {e}")
    
    print(f"\n🎉 Working voices: {', '.join(working_voices)}")
    return working_voices

# Utility function to clean existing scripts
def clean_existing_script(input_file: str, output_file: str = None):
    """Clean an existing script file by removing section headers."""
    if output_file is None:
        output_file = input_file.replace('.md', '_cleaned.md')
    
    content = Path(input_file).read_text(encoding="utf-8")
    cleaned_content = clean_script_content(content)
    
    Path(output_file).write_text(cleaned_content, encoding="utf-8")
    print(f"✅ Cleaned script saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Uncomment to test voices first
    # test_voices()
    
    # Option 1: Generate complete video content
    result = generate_video(
        topic="How to Build a Go-To-Market Strategy",
        voice_name="rachel",  # Try: rachel, domi, bella, antoni, etc.
        output_dir="gtm_strategy_video"
    )
    
    # Option 2: Generate audio from your own text
    # text_to_audio(
    #     text="Is your GTM strategy actually costing you millions? Delays mean rivals seize your market. You'll pay 2x, even 3x, more for customers. Hidden operational costs drain your budget. Win by leveraging proprietary data. Integrate AI for hyper-personalization. Align every team. Understand real customer shifts. Stop squandering resources. Build a winning GTM. Dominate your market.",
    #     output_file="custom_voiceover.mp3",
    #     voice_name="rachel"
    # )