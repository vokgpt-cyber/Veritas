# VERITAS 1.0: Ubuntu Server Deployment

Эта папка предназначена для IT-развертывания VERITAS 1.0 на Ubuntu-сервере с
NVIDIA GPU. Основной сценарий: RTX 4090 48 GB, Docker, pyannote Community-1,
GigaAM, vLLM/OpenAI-compatible LLM.

## Что находится в папке

- `docker-compose.server.yml` — серверный Docker Compose.
- `server.env.example` — шаблон настроек сервера.
- `START_VERITAS_UBUNTU.sh` — простой запуск.
- `STOP_VERITAS_UBUNTU.sh` — остановка.
- `START_VERITAS_UBUNTU.desktop` — запуск через GUI, если на сервере есть рабочий стол.
- `MODEL_GUIDE.md` — рекомендуемые LLM и настройки VRAM.
- `OPERATIONS_CHECKLIST.md` — короткий эксплуатационный чеклист для IT.

## Предварительные требования

На сервере должны быть установлены:

1. Ubuntu 22.04/24.04 LTS.
2. NVIDIA driver, проверяется командой `nvidia-smi`.
3. Docker Engine.
4. NVIDIA Container Toolkit.
5. Доступ к Hugging Face для первой загрузки pyannote и vLLM-моделей.

Для pyannote нужно принять условия модели:
`pyannote/speaker-diarization-community-1`, затем создать read-token.

## Получение кода из GitHub

Репозиторий:

```text
https://github.com/vokgpt-cyber/Veritas
```

Первое получение:

```bash
git clone https://github.com/vokgpt-cyber/Veritas.git
cd Veritas
```

Если репозиторий закрытый, GitHub сначала должен дать доступ нужным IT-аккаунтам.

Обновление уже установленной копии:

```bash
cd Veritas
git pull --ff-only
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

## Первый запуск

Из корня репозитория:

```bash
chmod +x deploy/it-ubuntu/*.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

Скрипт сам создаст `deploy/it-ubuntu/.env.server` с рандомными секретами.
Перед первой реальной обработкой IT должен открыть `.env.server` и прописать:

```bash
HF_TOKEN=hf_...
```

После запуска:

- локально на сервере: `http://localhost:5173`
- по сети: `http://SERVER_IP:5173`
- первичный вход: `admin / admin`

Пароль администратора нужно сменить перед пилотом.

## Запуск по двойному клику

Если на Ubuntu-сервере есть графическая оболочка:

1. Откройте папку `deploy/it-ubuntu`.
2. Разрешите запуск файлов как программ, если Ubuntu спросит.
3. Запустите `START_VERITAS_UBUNTU.desktop`.

На сервере без GUI используйте `START_VERITAS_UBUNTU.sh`.

## GPU и VRAM

Важный момент: Docker на GeForce/RTX обычно не отрезает фиксированные 30 GB
VRAM так же жестко, как обычную RAM. Для LLM это регулируется на уровне vLLM:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

Для RTX 4090 48 GB:

- 30 GB budget примерно равно `0.62`.
- 36 GB budget примерно равно `0.75`.
- 40 GB budget примерно равно `0.83`.

Если выбранная модель требует больше VRAM, чем выделено vLLM, она не должна
стартовать. Это правильное поведение: лучше явная ошибка, чем скрытое снижение
качества.

Проверка текущей памяти GPU:

```bash
nvidia-smi
```

Формула для `VLLM_GPU_MEMORY_UTILIZATION`:

```text
нужный_бюджет_GB / общий_VRAM_GB
```

Примеры для RTX 4090 48 GB:

```text
30 GB / 48 GB = 0.62
36 GB / 48 GB = 0.75
40 GB / 48 GB = 0.83
```

Если IT хочет выделить VERITAS 30 GB под LLM, нужно поставить:

```bash
VERITAS_GPU_VRAM_BUDGET_GB=30
VLLM_GPU_MEMORY_UTILIZATION=0.62
```

Если модель просит больше памяти, нужно выбрать один из трех честных вариантов:

- увеличить `VLLM_GPU_MEMORY_UTILIZATION`;
- уменьшить `VLLM_MAX_MODEL_LEN`;
- выбрать более легкую модель.

Не нужно включать скрытые fallback-модели или снижать качество pipeline.

## Переключение LLM

Базовый серверный режим использует vLLM:

```bash
COMPOSE_PROFILES=vllm
EPAM_SUMMARIZATION_PROVIDER=openai_compatible
EPAM_SUMMARIZATION_OPENAI_BASE_URL=http://vllm:8000/v1
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
```

Важно: в VERITAS 1.0 backend использует поле
`EPAM_SUMMARIZATION_OLLAMA_MODEL` как имя активной модели и для Ollama, и для
OpenAI-compatible vLLM. Поэтому при смене vLLM-модели меняйте оба поля:

```bash
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
```

После изменения модели:

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

Проверка, что vLLM видит модель:

```bash
curl http://127.0.0.1:8001/v1/models
```

Для временного отключения vLLM и использования внешнего Ollama:

```bash
COMPOSE_PROFILES=
EPAM_SUMMARIZATION_PROVIDER=ollama
EPAM_SUMMARIZATION_OLLAMA_BASE_URL=http://host.docker.internal:11434
EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b
```

## Обновление с GitHub

Обычный порядок обновления:

```bash
git pull
./deploy/it-ubuntu/START_VERITAS_UBUNTU.sh
```

Скрипт пересоберет backend/frontend images, не удаляя Docker volumes с данными,
кэшами моделей и настройками.

## Логи

```bash
docker compose --env-file deploy/it-ubuntu/.env.server \
  -f deploy/it-ubuntu/docker-compose.server.yml logs -f
```

Логи vLLM:

```bash
docker logs -f veritas-vllm
```

Логи backend:

```bash
docker logs -f veritas-backend
```

## Остановка

```bash
./deploy/it-ubuntu/STOP_VERITAS_UBUNTU.sh
```

Эта команда останавливает контейнеры, но не удаляет данные и кэши.

## Official references

- Docker resource constraints: `https://docs.docker.com/engine/containers/resource_constraints/`
- Docker GPU access: `https://docs.docker.com/desktop/features/gpu/`
- NVIDIA Container Toolkit: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/`
- vLLM OpenAI-compatible server: `https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html`
