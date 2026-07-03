"""Runtime configuration, loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model serving (vLLM OpenAI-compatible endpoint)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    temperature: float = 0.0

    # Browser (computer-use)
    viewport_width: int = 1280
    viewport_height: int = 800
    headless: bool = True
    # Path to a Chromium/Chrome binary; empty = use Playwright's own download
    chromium_executable: str = ""

    # Agent loop
    max_steps: int = 15
    max_screenshots: int = 3
    max_image_edge: int = 1280
    # "hybrid": numbered interactive elements overlaid on the screenshot and
    # element-id tools (efficient, reliable). "pixels": raw screenshots and
    # coordinate clicks only.
    agent_vision_mode: str = "hybrid"
    # What "computer use" controls: "browser" (safe default, Playwright) or
    # "desktop" (whole OS screen via pyautogui — install the [desktop] extra)
    computer_mode: str = "browser"

    # Voice mode
    stt_backend: str = "whisper"  # "whisper" (faster-whisper) or "moonshine"
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"  # auto / cuda / cpu
    tts_voice: str = "af_heart"  # Kokoro voice
    voice_narration: bool = True  # speak short progress lines during browse tasks

    runs_dir: str = "runs"

    # Long-term memory: a folder of plain markdown notes (Obsidian-compatible;
    # point MEMORY_DIR at an existing vault if you like)
    memory_enabled: bool = True
    memory_dir: str = "memory"
    # After each exchange, ask the model whether it learned a durable fact
    # about the user and save it as a note
    auto_memory: bool = True


settings = Settings()
