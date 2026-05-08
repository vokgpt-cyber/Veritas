# VERITAS 1.0: Note for EPAM IT

Коллеги, подготовлена пилотная версия VERITAS 1.0 для серверного развертывания
на Ubuntu + NVIDIA GPU.

## Repository

```text
https://github.com/vokgpt-cyber/Veritas
```

Нужен доступ к репозиторию для IT-аккаунтов GitHub.

## What to do

1. Подготовить Ubuntu-сервер с RTX 4090 48 GB.
2. Установить NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, Git.
3. Проверить GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

4. Клонировать репозиторий:

```bash
git clone https://github.com/vokgpt-cyber/Veritas.git
cd Veritas
```

5. Запустить:

```bash
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

6. Внести Hugging Face token в `deploy/it-ubuntu/.env.server`.
   Аккаунт должен иметь принятый доступ к:
   `pyannote/speaker-diarization-community-1`.

7. Перезапустить:

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## Key configuration files

- `deploy/it-ubuntu/README.md` — основная инструкция.
- `deploy/it-ubuntu/OPERATIONS_CHECKLIST.md` — эксплуатационный чеклист.
- `deploy/it-ubuntu/MODEL_GUIDE.md` — выбор LLM и параметры VRAM.
- `deploy/it-ubuntu/.env.server` — локальные серверные настройки, не коммитить.

## VRAM policy

Docker на RTX/GeForce обычно не выделяет жесткий VRAM-лимит контейнеру.
Для LLM лимит задается через vLLM:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

Для RTX 4090 48 GB:

- 30 GB: `0.62`
- 36 GB: `0.75`
- 40 GB: `0.83`

Если модель не помещается в VRAM, ожидаемое поведение — явная ошибка. Не нужно
включать скрытый fallback на более слабую модель только ради завершения pipeline.

## Default LLM server profile

Первый рекомендуемый серверный кандидат:

```bash
Qwen/Qwen3.6-27B
```

Поля модели в `.env.server` должны совпадать:

```bash
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
```

## Updates

```bash
cd Veritas
git pull --ff-only
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## Security

Не коммитить:

- `.env.server`;
- Hugging Face tokens;
- аудио;
- стенограммы;
- DOCX/JSON результатов;
- audit logs;
- model caches.

Пароль `admin / admin` нужно сменить перед пилотным использованием.
