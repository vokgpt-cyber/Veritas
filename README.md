# EPAM VERITAS

**Verbal Intelligence, Transcription and Summarization**

A powerful on-premise meeting transcription and intelligent protocol generation system powered by cutting-edge AI models.

## What It Does

- **Automatic Transcription**: Converts audio from meetings into accurate text transcripts using NeMo Conformer ASR
- **Speaker Identification**: Detects and labels multiple speakers in conversations using SpeechBrain diarization
- **Intelligent Summarization**: Generates structured meeting protocols with key topics, decisions, and action items using LLM-based summarization
- **Professional Document Generation**: Exports branded DOCX, JSON, and plain text protocols with EPAM formatting

## Technology Stack

**Core Infrastructure**
- FastAPI 0.110+
- Python 3.11
- CUDA 12.1 (GPU-accelerated)

**ML Components**
- NeMo Toolkit 1.23+ (Speech Recognition - Conformer-Transducer)
- SpeechBrain 1.0+ (Speaker Diarization - ECAPA-VOXCELEB)
- Qwen 2.5 7B Instruct (Text Summarization - 4-bit quantized)
- PyTorch 2.2+ with TorchAudio 2.2+

**Hardware Requirements**
- NVIDIA GPU with CUDA compute capability 7.5+ (RTX 4060 or better)
- 16GB VRAM minimum (8GB with 4-bit quantization)
- 50GB storage (for model cache)

## Quick Start

### Docker (Recommended)

```bash
# Clone repository
git clone <repository-url> EPAM-transcriber
cd EPAM-transcriber

# Create .env file with HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# Build and run
docker-compose up -d

# Access API at http://localhost:8000
# Documentation at http://localhost:8000/api/docs
```

### Manual Setup

**Windows:**
```cmd
cd scripts
start.bat
```

**Linux/Mac:**
```bash
cd scripts
bash start.sh
```

The script will automatically:
1. Create a Python virtual environment
2. Install dependencies from requirements.txt
3. Start the FastAPI server on http://127.0.0.1:8000

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Process Meeting
```bash
curl -X POST http://localhost:8000/api/meetings/upload \
  -F "audio=@meeting.mp3" \
  -F "topic=Project Planning Discussion"
```

### Get Results
```bash
curl http://localhost:8000/api/meetings/{meeting_id}/protocol
```

Full API documentation available at `/api/docs` (Swagger UI)

## Configuration

Edit `config/settings.yaml` to customize:
- Model selection and precision
- GPU device allocation
- ASR and diarization parameters
- Output formatting and branding
- Quality assurance thresholds

## Project Structure

```
EPAM-transcriber/
├── backend/
│   ├── app/          # FastAPI application
│   ├── core/         # Processing engines
│   ├── engine/       # ML model wrappers
│   └── utils/        # Helper utilities
├── config/           # Configuration files
├── data/             # Input/output directories
├── frontend/         # Web UI (React)
├── scripts/          # Start scripts
└── docker-compose.yml # Container orchestration
```

## License

Proprietary - EPAM Systems

## Support

For issues or questions, contact the development team or visit the internal documentation portal.
