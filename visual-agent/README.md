# 👁️ Visual Agent

A fully open-source **visual agentic AI** that runs on **48 GB VRAM** (local or rented):

- **Vision reasoning** — chat with a local vision-language model about images, screenshots, documents, charts.
- **Computer use (browser)** — give it a task and it drives a real browser: looks at the page, clicks numbered elements, types, scrolls, until done.
- **Computer use (desktop)** — control the *whole* OS screen: open LibreOffice Calc and edit a spreadsheet, open a photo in GIMP and edit it, use any app.
- **Voice mode** — talk to it like ChatGPT/Claude voice: it answers with a natural voice, you can **interrupt it mid-sentence**, and spoken commands ("open Wikipedia and…") run the computer-use agent with spoken progress.
- **Long-term memory** — it remembers you across sessions in a plain-markdown vault that is **Obsidian-compatible**: open the `memory/` folder in Obsidian to browse, edit, or graph what it knows.

**Stack (all open source):** [vLLM](https://github.com/vllm-project/vllm) serving [Qwen3-VL](https://huggingface.co/Qwen) · [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent loop · [Playwright](https://playwright.dev) browser · [pyautogui](https://github.com/asweigart/pyautogui)+[mss](https://github.com/BoboTiG/python-mss) desktop control · [FastRTC](https://github.com/gradio-app/fastrtc) (WebRTC + Silero VAD) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) STT · [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS · [Gradio](https://gradio.app) UI · Typer CLI.

```
 🎙️ Voice (WebRTC+VAD)──STT──┐                ┌──────────────────────────────┐
 🌐 Gradio UI ───────────────┼─▶ LangGraph ──▶│ vLLM · Qwen3-VL-32B · :8000  │
 ⌨️  CLI ────────────────────┘   agent loop   │ (local 48GB GPU or rented)   │
                                   │          └──────────────────────────────┘
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            Browser (Playwright)        Desktop (pyautogui)
            numbered elements,          open_app, click, drag,
            click/type/scroll           hotkeys, type — any app
```

## 1. Serve the model (vLLM)

On the machine with the GPU (yours **or rented**, see §5):

```bash
pip install vllm
./scripts/serve_vllm.sh     # Qwen/Qwen3-VL-32B-Instruct-FP8 on :8000
```

Model options for 48 GB (`MODEL_NAME=... ./scripts/serve_vllm.sh`):

| Model | VRAM (weights) | Notes |
|---|---|---|
| `Qwen/Qwen3-VL-32B-Instruct-FP8` (default) | ~33 GB | Best quality; strong native GUI grounding |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | ~31 GB | MoE — much faster, near-same quality |
| `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | ~19 GB | Biggest headroom / longest context |
| `ByteDance-Seed/UI-TARS-1.5-7B` | ~16 GB | GUI-specialist — great click accuracy, weaker chat |

2×24 GB cards: use the `--tensor-parallel-size 2` variant in the script.

## 2. Install the agent

```bash
cd visual-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,voice]"        # add ,desktop for OS-level control
playwright install chromium
cp .env.example .env
```

## 3. Use it

**Web UI** — vision chat + computer use + voice:

```bash
visual-agent ui                      # http://127.0.0.1:7860
```

- **Vision chat** tab: upload images, ask questions, streamed replies.
- **Computer use** tab: type a task, watch steps + live screenshots.
- **Voice** tab: click connect, talk. Kokoro speaks each sentence as the model
  generates it, and Silero VAD lets you barge in mid-reply — start talking and
  it stops, exactly like ChatGPT voice. Say a task ("go to wikipedia and…")
  and it runs the computer-use agent, narrating progress out loud.
  *(Mic access needs `localhost` or HTTPS.)*

**CLI:**

```bash
visual-agent chat -i photo.png                      # vision chat
visual-agent browse "find Iceland's population on wikipedia"
visual-agent desktop "open libreoffice calc, make a 3x3 multiplication table, save as /tmp/t.ods"
```

`HEADLESS=false` in `.env` shows the browser live. Desktop mode moves your
real mouse — slam the cursor into a screen corner to abort (pyautogui failsafe).
To make *voice* commands drive the desktop instead of the browser, set
`COMPUTER_MODE=desktop`.

## 4. Screenshots vs. faster grounding methods

Raw screenshots + pixel clicks work everywhere but are slow (every step ships a
big image) and clicks can miss. This project defaults to a better method:

| Method | How | Trade-off |
|---|---|---|
| Raw screenshots (`AGENT_VISION_MODE=pixels`) | model guesses pixel coords | works on anything, least reliable |
| **Set-of-Marks hybrid (default)** | interactive elements extracted from the DOM, numbered on the screenshot + listed as text; model calls `click_element(7)` | ~exact clicks, fewer tokens, fewer retries |
| Pure accessibility tree (no images) | text-only element list | cheapest, but blind to layout/canvas/images |

The hybrid keeps the screenshot (so the model still *sees* the page) but acts
through numbered elements — clicks land on the element center computed from
the real DOM, not from visual guessing. Desktop mode has no DOM, so it uses
pixel grounding with coordinate scaling.

## 5. No GPU? Rent one and run from your laptop

Only the **model server** needs the GPU. Everything else — browser, desktop
control, voice, UI, CLI — runs on your laptop (CPU is fine).

1. Rent a 48 GB GPU (RunPod / Vast.ai / Lambda: A6000, L40S, A40 are cheap).
2. On the rented box: `pip install vllm && ./scripts/serve_vllm.sh`
3. On your laptop, tunnel the port:
   ```bash
   ssh -N -L 8000:localhost:8000 user@rented-box
   ```
4. Keep `VLLM_BASE_URL=http://localhost:8000/v1` in your local `.env`, then run
   `visual-agent ui` / `browse` / `desktop` locally as usual.

For voice on a CPU-only laptop set `WHISPER_MODEL=small` (or
`STT_BACKEND=moonshine`); Kokoro TTS is fast on CPU already.

## 6. Long-term memory (Obsidian-compatible)

The agent remembers you across sessions — shared by chat, voice, and the CLI:

- After each exchange, one background model call decides whether it learned a
  **durable fact** ("user's name is Varun", "prefers LibreOffice") and saves it
  as a small markdown note in `memory/notes/`. Full conversations are logged
  to `memory/transcripts/`.
- At the start of every reply, relevant + recent notes are injected into the
  model's context, so "you remember my dog's name?" just works.
- The vault is **plain markdown with YAML frontmatter** — open `memory/` as an
  Obsidian vault to browse, edit, delete, or graph it. Set `MEMORY_DIR` to an
  existing vault to share it.
- Manage it anywhere: the **Memory** tab in the UI (add/forget/refresh),
  `visual-agent memory` / `visual-agent memory --forget <file>` in the
  terminal, or just edit the files.
- Kill switches: `AUTO_MEMORY=false` (no automatic saving) or
  `MEMORY_ENABLED=false` (no memory at all).

Recall is transparent keyword matching over your notes — no database, no
embedding model, nothing to babysit. (If you ever outgrow it, graph-based
memory engines like [Graphiti](https://github.com/getzep/graphiti) are the
heavyweight open-source alternative, at the cost of running Neo4j.)

## How the agent works

`src/visual_agent/agent.py` builds a LangGraph state machine shared by browser
and desktop modes:

1. **agent** node — the VLM gets the task + latest annotated screenshot and
   emits one tool call (ReAct: describe what it sees, then act).
2. **tools** node — the action runs (Playwright or pyautogui), a fresh
   observation is captured into `runs/<timestamp>/`, old screenshots are pruned
   from context (`MAX_SCREENSHOTS`).
3. Loop until `finish(answer)` or `MAX_STEPS`.

Voice (`src/visual_agent/voice.py`): FastRTC streams mic audio; on a pause,
Whisper transcribes, a one-word LLM router picks chat vs. task, replies are
chunked into sentences and spoken by Kokoro as tokens stream. Interruptions
close the generator, cancelling speech and generation.

## Tests (no GPU, no mic, no display needed)

```bash
pytest    # 17 tests: agent loop + Set-of-Marks vs real headless Chromium,
          # desktop coordinate scaling (mocked), voice chunking/routing
```

## Safety notes

- The browser agent controls a real browser; the **desktop agent controls your
  real mouse and keyboard** — prefer running it in a VM or a spare desktop
  session, review tasks first, keep `MAX_STEPS` modest, and never leave it
  logged in to sensitive accounts. Corner-slam the mouse to abort.
- Headless servers can still run desktop mode under a virtual display:
  `sudo apt install xvfb && xvfb-run -s "-screen 0 1280x800x24" visual-agent desktop "..."`.
