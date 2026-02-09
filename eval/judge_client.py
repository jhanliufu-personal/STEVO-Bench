# eval/judge_client.py
"""
Judge clients for VLM-based evaluation.

Supports:
  - OpenAI
  - Anthropic
  - Google Gemini


  Usage (typical):
  client = make_judge_client(provider="openai", model="gpt-4o-mini")
  text = client.judge(prompt, init_image_path, final_image_path)

Env vars:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY   (Gemini)
"""

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol

from PIL import Image

# -----------------------------
# Helpers
# -----------------------------

def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def _image_to_data_url(image_path: Path) -> str:
    """
    Returns a data URL: data:<mime>;base64,<...>
    OpenAI accepts data URLs for image_url inputs.
    """
    b = image_path.read_bytes()
    mime = _guess_mime(image_path)
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"


def _image_to_base64(image_path: Path) -> tuple[str, str]:
    """
    Returns (mime_type, base64_string)
    Anthropic/Gemini accept base64 payloads.
    """
    b = image_path.read_bytes()
    mime = _guess_mime(image_path)
    return mime, base64.b64encode(b).decode("utf-8")


def _ensure_exists(p: Path) -> Path:
    p = Path(p).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


# -----------------------------
# Interface
# -----------------------------

class JudgeClient(Protocol):
    provider: str
    model: str

    def judge(self, prompt: str, init_image: Path, final_image: Path) -> str:
        """Return raw text response from the judge model."""
        # Includes both initial and final frame because some judge questions involve comparisons
        ...


# -----------------------------
# OpenAI
# -----------------------------

@dataclass
class OpenAIJudgeClient:
    model: str
    provider: str = "openai"

    def __post_init__(self) -> None:
        from openai import OpenAI  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        self._client = OpenAI(api_key=api_key)

    def judge(self, prompt: str, init_image: Path, final_image: Path) -> str:
        init_image = _ensure_exists(init_image)
        final_image = _ensure_exists(final_image)

        init_url = _image_to_data_url(init_image)
        final_url = _image_to_data_url(final_image)

        # Standard vision chat format: text + images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": init_url}},
                    {"type": "image_url", "image_url": {"url": final_url}},
                ],
            }
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


# -----------------------------
# Anthropic
# -----------------------------

@dataclass
class AnthropicJudgeClient:
    model: str
    provider: str = "anthropic"

    def __post_init__(self) -> None:
        import anthropic  # type: ignore
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable.")
        self._client = anthropic.Anthropic(api_key=api_key)

    def judge(self, prompt: str, init_image: Path, final_image: Path) -> str:
        init_image = _ensure_exists(init_image)
        final_image = _ensure_exists(final_image)

        init_mime, init_b64 = _image_to_base64(init_image)
        final_mime, final_b64 = _image_to_base64(final_image)

        # Anthropic messages: content blocks including images
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "source": {"type": "base64", "media_type": init_mime, "data": init_b64}},
            {"type": "image", "source": {"type": "base64", "media_type": final_mime, "data": final_b64}},
        ]

        msg = self._client.messages.create(
            model=self.model,
            max_tokens=800,  # modest default; adjust as needed
            messages=[{"role": "user", "content": content}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        out_parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                out_parts.append(block.text)
        return "".join(out_parts).strip()


# -----------------------------
# Gemini
# -----------------------------

@dataclass
class GeminiJudgeClient:
    model: str
    provider: str = "gemini"

    def __post_init__(self) -> None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
        self._client = genai.Client(api_key=api_key)

    def judge(self, prompt: str, init_image: Path, final_image: Path) -> str:
        init_image = _ensure_exists(init_image)
        final_image = _ensure_exists(final_image)

        # Gemini accepts PIL images directly
        init_pil = Image.open(init_image).convert("RGB")
        final_pil = Image.open(final_image).convert("RGB")

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[{"text": prompt}, init_pil, final_pil],
        )
        return getattr(resp, "text", "") or ""


# -----------------------------
# Factory
# -----------------------------

ProviderName = Literal["openai", "anthropic", "gemini"]


def make_judge_client(provider: ProviderName, model: str) -> JudgeClient:
    """
    Simple factory.
    Example:
      client = make_judge_client("openai", "gpt-4o-mini")
    """
    provider = provider.lower().strip()  # type: ignore

    if provider == "openai":
        return OpenAIJudgeClient(model=model)
    if provider == "anthropic":
        return AnthropicJudgeClient(model=model)
    if provider == "gemini":
        return GeminiJudgeClient(model=model)

    raise ValueError(f"Unknown provider: {provider}")
