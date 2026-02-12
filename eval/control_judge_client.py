# eval/control_judge_client.py
"""
Gemini-only control judge client that attaches the FULL video.

Env:
  GOOGLE_API_KEY

Deps:
  pip install google-genai
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ControlJudgeClient(Protocol):
    provider: str
    model: str

    def judge(self, prompt: str, video_path: Path) -> str:
        ...


@dataclass
class GeminiControlJudgeClient:
    model: str = "gemini-3-pro-preview"
    provider: str = "gemini"

    def __post_init__(self) -> None:
        from google import genai  # type: ignore

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
        self._client = genai.Client(api_key=api_key)

    # def judge(self, prompt: str, video_path: Path) -> str:
    #     from google.genai import types  # type: ignore

    #     video_path = Path(video_path).expanduser().resolve()
    #     if not video_path.exists():
    #         raise FileNotFoundError(video_path)

    #     # Upload the full video, then reference it in the request.
    #     uploaded = self._client.files.upload(file=str(video_path))

    #     resp = self._client.models.generate_content(
    #         model=self.model,
    #         contents=[
    #             {"text": prompt},
    #             types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4"),
    #         ],
    #     )
    #     return getattr(resp, "text", "") or ""

    def judge(self, prompt: str, video_path: Path) -> str:
        from google.genai import types
        import time

        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        uploaded = self._client.files.upload(file=str(video_path))

        # 🔥 Wait until file is ACTIVE
        while True:
            file_info = self._client.files.get(name=uploaded.name)
            if file_info.state == "ACTIVE":
                break
            elif file_info.state == "FAILED":
                raise RuntimeError("File processing failed.")
            # print('Video is still inactive')
            time.sleep(2)

        # print('Here we go!')        
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                {"text": prompt},
                types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4"),
            ],
        )

        return getattr(resp, "text", "") or ""



def make_control_judge_client(model: str = "gemini-3-pro-preview") -> ControlJudgeClient:
    return GeminiControlJudgeClient(model=model)
