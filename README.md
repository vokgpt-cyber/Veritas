# EPAM VERITAS

**Verbal Intelligence, Transcription and Summarization**

Version: **1.0.0 pilot baseline**

On-premise transcription, court hearing stenograms, administrative meeting
protocols, and action-list drafting for EPAM law firm workflows.

## Deployment Entry Points

- Windows workstation: `START_VERITAS_DOCKER.bat`
- Ubuntu server for IT: `deploy/it-ubuntu/README.md`
- Short IT handoff note: `IT_NOTE_FOR_EPAM_IT.md`
- Release notes: `RELEASE_1_0.md`

## What It Does

- **Automatic Transcription**: Russian-first transcription with GigaAM.
- **Speaker Diarization**: pyannote `speaker-diarization-community-1`.
- **Court Hearing Stenograms**: exact speaker-count controls, case dictionary,
  speaker review, DOCX export.
- **Administrative Protocols**: participants, accepted decisions, working theses,
  action items, citations, and action-list DOCX export.
- **Developer Settings**: admin/developer-only model and prompt controls for
  controlled experiments.

## Technology Stack

**Core Infrastructure**
- FastAPI
- Python 3.11
- Docker
- CUDA/NVIDIA GPU acceleration

**ML Components**
- GigaAM v3 for Russian ASR
- pyannote Community-1 for diarization
- Gemma 4 26B via Ollama on Windows workstation
- vLLM/OpenAI-compatible LLM support for Ubuntu server experiments
- React + Vite frontend

**Hardware Requirements**
- Local workstation: RTX 3090 24 GB target.
- IT server: RTX 4090 48 GB target.
- Storage: allow large model caches.

## Quick Start

### Windows Docker

```cmd
START_VERITAS_DOCKER.bat
```

Open:

```text
http://localhost:5173
```

Default login:

```text
admin / admin
```

Change the admin password before pilot use.

### Ubuntu Server

```bash
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

See `deploy/it-ubuntu/README.md`.

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
