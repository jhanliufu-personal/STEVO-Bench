import base64
import io
import json
import re
import yaml  
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from PIL import Image
from google import genai

# -----------------------------
# YAML IO (preserve readability)
# -----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    # Keep output pretty and stable for diffs
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            width=100,
            default_flow_style=False,
        )

# -----------------------------
# Hashing / digests
# -----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

# -----------------------------
# Path resolution
# -----------------------------

def _find_task_yaml_in_dir(task_dir: Path) -> Path:
    """By convention <task_dir>/<task_dir_name>.yaml"""
    folder = task_dir.name
    y = task_dir / f"{folder}.yaml"
    if not y.exists():
        raise FileNotFoundError(f"Missing task YAML: {y}")
    return y

def _find_init_frame_in_dir(task_dir: Path) -> Path:
    """By convention <task_dir>/<task_dir_name>_init_frame.png"""
    folder = task_dir.name
    img = task_dir / f"{folder}_init_frame.png"
    if not img.exists():
        raise FileNotFoundError(f"Missing init frame: {img}")
    return img

def resolve_task_paths(task_dir: Path) -> Tuple[Path, Path, str, Path]:
    """
    Expects task_dir to be a directory.

    Convention:
      <task_dir>/<task_dir_name>.yaml
      <task_dir>/<task_dir_name>_init_frame.png

    Returns:
      task_dir, task_yaml_path, folder_name, init_frame_path
    """
    task_dir = Path(task_dir).expanduser().resolve()
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)
    if not task_dir.is_dir():
        raise ValueError(f"Expected a task directory, got file: {task_dir}")

    folder_name = task_dir.name
    task_yaml_path = task_dir / f"{folder_name}.yaml"
    init_frame_path = task_dir / f"{folder_name}_init_frame.png"

    if not task_yaml_path.exists():
        raise FileNotFoundError(f"Task YAML not found: {task_yaml_path}")
    if not init_frame_path.exists():
        # raise FileNotFoundError(f"Init frame not found: {init_frame_path}")
        # print(f"Init frame not found: {init_frame_path}")
        init_frame_path = None

    return task_dir, task_yaml_path, folder_name, init_frame_path


# -----------------------------
# Image handling
# -----------------------------

def image_to_inline_data(image_path: Path, max_side: int = 1024) -> Dict[str, Any]:
    """Convert image to inline data for Gemini API."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"inline_data": {"mime_type": "image/png", "data": data}}


# -----------------------------
# Gemini API
# -----------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from Gemini response (handles markdown code blocks)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find JSON object in Gemini response.")
    return json.loads(m.group(0))


def call_nanobanana(
    api_key: str,
    model: str,
    prompt: str,
    output_path: Path,
    input_image: Path = None,
    overwrite: bool = False
) -> Optional[Path]:
    # 1. Validate mandatory string inputs
    if not all(isinstance(i, str) and i.strip() for i in [api_key, model, prompt]):
        raise ValueError("api_key, model, and prompt must be non-empty strings.")

    # 2. Handle Path resolution
    output_path = Path(output_path).expanduser().resolve()
    
    if output_path.exists() and not overwrite:
        print(
            f"Output already exists: {output_path}. Set overwrite=True to replace it."
        )
        return None

    # 3. Prepare content list for the model
    contents = [prompt]
    
    if input_image:
        input_image_path = Path(input_image).expanduser().resolve()
        if not input_image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        # Load and add image to contents
        image = Image.open(input_image_path)
        contents.append(image)

    # 4. Initialize client and generate
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    if not response.parts:
        raise RuntimeError("Gemini response contained no parts.")

    # 5. Process response parts
    image_saved = False
    for part in response.parts:
        if part.text:
            print(f"Model response: {part.text}")
        elif part.inline_data:
            # Note: as_image() assumes the part contains image data
            generated_img = part.as_image()
            generated_img.save(output_path)
            image_saved = True

    if not image_saved:
        print("Warning: No image data was returned in the response parts.")

    return output_path