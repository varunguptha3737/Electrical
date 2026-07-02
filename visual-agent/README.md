# 👁️ Visual Agent

A fully open-source **visual agentic AI** that runs on a single **48 GB VRAM** machine:

- **Vision reasoning** — chat with a local vision-language model about images, screenshots, documents, charts.
- **Computer use** — give it a task and it drives a real browser from screenshots: it looks at the page, decides, clicks at pixel coordinates, types, scrolls, until the task is done.

**Stack (all open source):** [vLLM](https://github.com/vllm-project/vllm) serving [Qwen3-VL](https://huggingface.co/Qwen) · [LangGraph](https://github.com/langchain-ai/langgraph) agent loop (ReAct-style) · [Playwright](https://playwright.dev) browser control · [Gradio](https://gradio.app) web UI · Typer CLI.

```
Gradio UI ─┐                       ┌──────────────────────────────┐
           ├─▶ LangGraph agent ──▶ │ vLLM · Qwen3-VL-32B · :8000  │
CLI ───────┘        │              └──────────────────────────────┘
                    ▼
        Playwright browser tools
        navigate · click(x,y) · type · scroll · keys
        (fresh screenshot fed back to the model every step)
```

## 1. Requirements

- NVIDIA GPU(s) with ~48 GB total VRAM (1× RTX 6000 Ada / L40S / A6000, or 2× 3090/4090)
- Python 3.10+, CUDA drivers
- ~40 GB disk for model weights

## 2. Serve the model (vLLM)

```bash
pip install vllm                # ideally in its own venv
./scripts/serve_vllm.sh         # serves Qwen/Qwen3-VL-32B-Instruct-FP8 on :8000
```

Model options for 48 GB (set `MODEL_NAME` env var, and in `.env`):

| Model | VRAM (weights) | Notes |
|---|---|---|
| `Qwen/Qwen3-VL-32B-Instruct-FP8` (default) | ~33 GB | Best quality; strong native GUI grounding |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | ~31 GB | MoE — much faster, near-same quality |
| `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | ~19 GB | Biggest headroom / longest context |
| `ByteDance-Seed/UI-TARS-1.5-7B` | ~16 GB | Specialized GUI-agent model — great click accuracy, weaker general chat |

For 2×24 GB cards, use the `--tensor-parallel-size 2` variant in `scripts/serve_vllm.sh`.

## 3. Install the agent

```bash
cd visual-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
cp .env.example .env            # adjust if your vLLM host/model differ
```

## 4. Use it

**Web UI** (vision chat + computer-use with live screenshots):

```bash
visual-agent ui                 # http://127.0.0.1:7860
```

**CLI:**

```bash
# vision chat (attach images to your first message)
visual-agent chat -i photo.png

# autonomous browser task — step screenshots land in ./runs/<timestamp>/
visual-agent browse "Go to en.wikipedia.org and find the population of Iceland"
```

Set `HEADLESS=false` in `.env` to watch the agent drive the browser live.

**Python API:** see `examples/analyze_chart.py` and `examples/browse_task.py`.

## How the agent works

`src/visual_agent/agent.py` builds a LangGraph state machine:

1. **agent** node — the VLM gets the task + the latest screenshot and emits one tool call (ReAct: it states what it sees/plans, then acts).
2. **tools** node — the call runs in Playwright (fixed 1280×800 viewport, so model pixel coordinates map 1:1 to mouse clicks), a fresh screenshot is captured, saved to `runs/`, and appended to the conversation. Screenshots older than `MAX_SCREENSHOTS` are pruned from context.
3. Loop until the model calls `finish(answer)` or `MAX_STEPS` is hit.

## Tests (no GPU needed)

```bash
pytest        # scripted fake LLM + real headless Chromium on a local test page
```

## Extensions (out of scope here)

- OS-level desktop control (needs a sandboxed VM — browser-only is safer/simpler)
- Pointing [Open WebUI](https://github.com/open-webui/open-webui) at the same vLLM endpoint for a richer plain-chat frontend
- Multi-agent orchestration, RAG over documents, fine-tuning

## Safety note

The computer-use agent controls a real browser. Run it against sites you trust,
review tasks before running, and keep `MAX_STEPS` modest. Don't give it
sessions that are logged in to sensitive accounts.
