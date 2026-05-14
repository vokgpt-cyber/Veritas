# VERITAS 1.0: note for EPAM IT

Коллеги, подготовлена пилотная версия VERITAS 1.0 для серверного развертывания
на Ubuntu + NVIDIA GPU. Перед передачей пилотным пользователям просьба пройти
не только эксплуатационную, но и security-проверку.

## Repository

```text
https://github.com/vokgpt-cyber/Veritas
```

Рекомендуемая ветка для серверного пакета:

```bash
git clone --branch release/veritas-1.0-it https://github.com/vokgpt-cyber/Veritas.git
cd Veritas
```

## What to do

1. Подготовить Ubuntu 22.04/24.04 LTS сервер с RTX 4090 48 GB.
2. Установить NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, Git.
3. Проверить GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

4. Запустить VERITAS:

```bash
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

5. Внести Hugging Face read-token в `deploy/it-ubuntu/.env.server`.
   Аккаунт должен иметь доступ к `pyannote/speaker-diarization-community-1`.

6. Перезапустить:

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## Key files

- `deploy/it-ubuntu/README.md` - основная инструкция.
- `deploy/it-ubuntu/OPERATIONS_CHECKLIST.md` - эксплуатационный чеклист.
- `deploy/it-ubuntu/SECURITY_REVIEW.md` - security-чеклист перед пилотом.
- `deploy/it-ubuntu/MODEL_GUIDE.md` - выбор LLM и параметры VRAM.
- `deploy/it-ubuntu/.env.server` - локальные серверные настройки, не коммитить.

## Security gate

Перед подключением пилотных пользователей:

- Web UI должен оставаться на `127.0.0.1` до публикации через утвержденный
  reverse proxy, VPN или firewall.
- Backend `8765`, Ollama `11435` и vLLM `8001` не публиковать в LAN напрямую.
- Сменить bootstrap admin password. Первый запуск генерирует случайный пароль и
  показывает его в терминале.
- Для сетевого режима предпочтительно заранее задать
  `EPAM_AUTH_DEFAULT_PASSWORD_HASH`.
- Не включать vLLM `--trust-remote-code`, пока конкретный model repository не
  пройдет review как исполняемый код.
- Не использовать Docker images с `:latest`; закреплять tag или digest.
- Просканировать Docker images:

```bash
docker scout cves epam-veritas-backend:1.0
docker scout cves epam-veritas-frontend:1.0
docker scout cves ollama/ollama:0.13.4
docker scout cves vllm/vllm-openai:v0.18.2
```

Если Docker Scout недоступен:

```bash
trivy image epam-veritas-backend:1.0
trivy image epam-veritas-frontend:1.0
trivy image ollama/ollama:0.13.4
trivy image vllm/vllm-openai:v0.18.2
```

## VRAM policy

Основной выверенный LLM baseline для пилота - Gemma 4 через Ollama:

```bash
COMPOSE_PROFILES=
EPAM_SUMMARIZATION_PROVIDER=ollama
EPAM_SUMMARIZATION_OLLAMA_BASE_URL=http://ollama:11434
EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b
OLLAMA_IMAGE=ollama/ollama:0.13.4
VERITAS_READ_ONLY_ROOTFS=false
```

Qwen/Qwen3.6-27B через vLLM - только первый альтернативный кандидат для будущих
экспериментов, не baseline для первичного развертывания.

Docker на RTX/GeForce обычно не отрезает жесткий VRAM-лимит контейнеру. Для
будущих vLLM-экспериментов лимит задается через:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

Для RTX 4090 48 GB:

- 30 GB: `0.62`
- 36 GB: `0.75`
- 40 GB: `0.83`

Если модель не помещается в VRAM, ожидаемое поведение - явная ошибка. Не нужно
включать скрытый fallback на более слабую модель только ради завершения pipeline.

## Data handling

Не коммитить и не передавать через Git:

- `.env.server`;
- Hugging Face tokens;
- аудио и видео;
- стенограммы;
- DOCX/JSON результатов;
- audit logs;
- model caches.

VERITAS должен обрабатывать реальные материалы только on-premise. Cloud LLM
endpoints для реальных данных не использовать.
