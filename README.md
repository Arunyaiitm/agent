# Local Agent

A small command-line AI agent that runs entirely on your computer through Ollama.

It can:

- Chat with you in the terminal
- List files in this project
- Read files in this project
- Write files in this project when you ask it to

## Setup

Install Ollama from [ollama.com/download](https://ollama.com/download), open it, and download a local coding model:

```bash
ollama pull qwen2.5-coder:3b
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

No API key is required. The optional `.env` file only controls the local Ollama model:

```bash
LOCAL_MODEL=qwen2.5-coder:3b
OLLAMA_URL=http://127.0.0.1:11434
```

## Run

```bash
local-agent
```

Or:

```bash
python -m local_agent
```

Type `exit` or `quit` to stop.

## Notes

The agent is restricted to the project folder by default. That means its file tools cannot read or write outside this repository.

The model runs locally. The first response may take a little longer while the model loads into memory.
