# eval/judge_client.py
"""
VLM judge client supporting Gemini and OpenAI.

Env vars:
  GOOGLE_API_KEY  — required for Gemini models
  OPENAI_API_KEY  — required for OpenAI models

Deps:
  pip install google-genai          # Gemini
  pip install openai opencv-python  # OpenAI (opencv used for video frame sampling)
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


class ControlJudgeClient(Protocol):
    provider: str
    model: str

    def judge(self, prompt: str, video_path: Path) -> str: ...
    def judge_image(self, prompt: str, image_path: Path) -> str: ...


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

@dataclass
class GeminiControlJudgeClient:
    model: str = "gemini-3.1-pro-preview"
    provider: str = "gemini"

    def __post_init__(self) -> None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
        self._client = genai.Client(api_key=api_key)
        print(f"Using GOOGLE_API_KEY")

    def judge(self, prompt: str, video_path: Path) -> str:
        from google.genai import types  # type: ignore
        import time

        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"  [gemini] uploading {video_path.name} ...")
        uploaded = self._client.files.upload(file=str(video_path))
        while True:
            file_info = self._client.files.get(name=uploaded.name)
            if file_info.state == "ACTIVE":
                break
            elif file_info.state == "FAILED":
                raise RuntimeError("File processing failed.")
            time.sleep(2)

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                {"text": prompt},
                types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4"),
            ],
        )
        return getattr(resp, "text", "") or ""

    def judge_image(self, prompt: str, image_path: Path) -> str:
        from google.genai import types  # type: ignore

        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
        img_bytes = image_path.read_bytes()

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=mime),
                {"text": prompt},
            ],
        )
        return getattr(resp, "text", "") or ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@dataclass
class OpenAIControlJudgeClient:
    model: str = "gpt-4o"
    provider: str = "openai"
    n_frames: int = 64  # frames sampled from the video for judging

    def __post_init__(self) -> None:
        import openai  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        self._client = openai.OpenAI(api_key=api_key)
        print(f"Using OPENAI_API_KEY")

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    def _sample_frames_b64(self, video_path: Path) -> list[str]:
        """Sample self.n_frames uniformly from video; return base64-encoded JPEG strings."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for OpenAI video judging: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.n_frames
        n = min(self.n_frames, total)
        indices = [int(i * total / n) for i in range(n)]

        frames: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(self._encode_bytes(buf.tobytes()))
        cap.release()
        return frames

    def judge(self, prompt: str, video_path: Path) -> str:
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"  [openai] sampling {self.n_frames} frames from {video_path.name} ...")
        frames_b64 = self._sample_frames_b64(video_path)
        content: list = [{"type": "text", "text": prompt}]
        for f_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f_b64}", "detail": "low"},
            })

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def judge_image(self, prompt: str, image_path: Path) -> str:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
        img_b64 = self._encode_bytes(image_path.read_bytes())

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")
_GEMINI_PREFIXES = ("gemini",)


def make_judge_client(
    model: str,
    provider: Optional[str] = None,
) -> ControlJudgeClient:
    """
    Instantiate the right judge client for (provider, model).

    provider: "gemini" | "openai" | None (auto-detect from model name prefix)
    """
    if provider is None:
        m = model.lower()
        if any(m.startswith(p) for p in _OPENAI_PREFIXES):
            provider = "openai"
        else:
            provider = "gemini"

    if provider == "openai":
        return OpenAIControlJudgeClient(model=model)
    elif provider == "gemini":
        return GeminiControlJudgeClient(model=model)
    else:
        raise ValueError(f"Unknown provider {provider!r}. Supported: 'gemini', 'openai'")
