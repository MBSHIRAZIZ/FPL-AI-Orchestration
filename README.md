# FPL AI Orchestration

**CM3070 Final Year Project**

A multi-modal AI orchestration system that combines audio transcription, social media sentiment analysis, and statistical prediction to generate evidence-based Fantasy Premier League recommendations.

## Project Template

Based on the CM3020 "Orchestrating AI Models to Achieve a Goal" template (Project Number 4.1).

## Architecture
Input Layer 
-Audio (podcast)
-Social media (reddit)
-Tabular Data (FPL API)

Processing Layer
-Model 1 : Whisper -> Llama 3.1 (summarisation)
-Model 2 : VADER + FPL keyword override (sentiment)
-Model 3 : Linear Regression (points prediction)

Orchestration Layer
-Python Orchestrator which combines output, applies rules

Output Layer
-Flask Web Dashboard (Per-Player recommendations)

## Setup

### Prerequisites

- Python 3.8+
- ffmpeg (for Whisper audio processing)
- Ollama (for local LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/mbshiraziz/FPL-AI-Orchestration.git
cd FPL-AI-Orchestration

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.1

# Running the pipeline
python prototype_fpl.py

# Running the web dashboard
python app.py
# Open http://127.0.0.1:5000

# Running Tests
pytest tests.py -v
