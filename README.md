# Video Script Generator with AI Voiceover

This project is an automated tool that generates video scripts using Google's Gemini AI and converts them into natural-sounding voiceovers using text-to-speech technology.

## Features

- 🤖 AI-powered script generation using Google's Gemini AI
- 🎯 Generates properly formatted video scripts with timestamps and scene directions
- 🗣️ Converts scripts to natural-sounding voiceovers using Coqui TTS
- ⚡ Simple and efficient workflow

## Prerequisites

- Python 3.10 or higher
- Google API Key for Gemini AI
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Google API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```python
from test import generate_video

# Generate a video script and voiceover
result = generate_video("How to Build a Go-To-Market Strategy")
print("Generated Script:", result["script"])
print("Voiceover MP3:", result["voiceover"])
```

### Individual Components

#### Generate Script Only

```python
from test import generate_script

script_file = generate_script("Your Topic", "output_script.md")
```

#### Generate Voiceover Only

```python
from test import generate_voiceover

mp3_file = generate_voiceover(
    script_file="your_script.md",
    output_file="output.mp3",
    voice_model="tts_models/en/ljspeech/tacotron2-DDC"
)
```

## Project Structure

```
.
├── test.py           # Main script with core functionality
├── requirements.txt  # Project dependencies
├── .env             # Environment variables (create this)
└── README.md        # Project documentation
```

## How It Works

1. **Script Generation**:

   - Uses Gemini AI through LangChain to generate video scripts
   - Scripts include timestamps and scene directions
   - Format: `**(Scene: ...)** **(0-5 seconds):** narration...`

2. **Voice Generation**:
   - Uses Coqui TTS for high-quality voice synthesis
   - Automatically cleans script by removing scene directions and timestamps
   - Generates MP3 file with natural-sounding voiceover

## Dependencies

### Text-to-Speech

- TTS >= 0.15.0 (Coqui TTS)
- torch >= 2.0.0
- torchaudio >= 2.0.0
- numpy >= 1.22.0

### AI and Language Processing

- langchain-google-genai >= 0.0.5
- langchain >= 0.1.0

### Utilities

- python-dotenv >= 0.19.0
- pathlib >= 1.0.1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **TTS Installation Issues**

   - Make sure you have the correct Python version
   - Install PyTorch before installing TTS

2. **API Key Errors**

   - Verify your Gemini API key is correctly set in the .env file
   - Check if the .env file is in the correct location

3. **Voice Model Issues**
   - The default model "tts_models/en/ljspeech/tacotron2-DDC" should download automatically
   - If issues persist, try downloading the model manually

### Getting Help

If you encounter any issues:

1. Check the error message carefully
2. Verify all dependencies are installed correctly
3. Make sure your Python version is compatible
4. Check if your API key is valid and properly set

## Future Improvements

- [ ] Add support for multiple languages
- [ ] Implement different voice models and styles
- [ ] Add background music integration
- [ ] Create a web interface
- [ ] Add video generation capabilities
