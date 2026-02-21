import os
import time
from pathlib import Path
from typing import Optional

from .base import WorldModelRunner


class VeoRunner(WorldModelRunner):
    """World-model adapter for Google Veo via the google-genai SDK.

    Submits an async video-generation job, polls until completion, then
    downloads the resulting MP4 to output_path.

    Config keys (in addition to base keys):
        api_key_env    (str)  Name of the environment variable holding the API key.
                              Default: "GOOGLE_API_KEY".
        model_id       (str)  Veo model identifier.
                              Default: "veo-3-0-generate-001".
        poll_interval  (int)  Seconds to wait between polling attempts. Default: 10.
        timeout        (int)  Max seconds to wait for a single generation. Default: 600.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        api_key_env = config.get("api_key_env", "GOOGLE_API_KEY")
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"[{name}] Missing required environment variable: {api_key_env}"
            )
        self.model_id: str = config.get("model_id", "veo-3-0-generate-001")
        self.poll_interval: int = int(config.get("poll_interval", 10))
        self.timeout: int = int(config.get("timeout", 600))

    def generate(
        self,
        task_id: str,
        prompt: str,
        init_frame: Optional[Path],
        output_path: Path,
        camera_control: Optional[str] = None,
    ) -> bool:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Build optional conditioning image
            image = None
            if init_frame and init_frame.exists():
                image = types.Image.from_file(str(init_frame))

            # Submit async generation job
            operation = client.models.generate_videos(
                model=self.model_id,
                prompt=prompt,
                image=image,
                config=types.GenerateVideoConfig(number_of_videos=1),
            )

            # Poll until done or timeout
            elapsed = 0
            while not operation.done:
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                operation = client.operations.get(operation)
                if elapsed >= self.timeout:
                    print(
                        f"[{self.name}] Timeout ({elapsed}s) waiting for task {task_id}"
                    )
                    return False

            videos = operation.response.generated_videos
            if not videos:
                print(f"[{self.name}] No videos returned for task {task_id}")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            client.files.download(
                file=videos[0].video, download_path=str(output_path)
            )
            return True

        except Exception as e:
            print(f"[{self.name}] Error generating task {task_id}: {e}")
            return False
