"""
Stage 7 — Cloud Quality Gate
When the local quality score for a house's best frame is below
config.QUALITY_THRESHOLD, send the top-N candidate frames to Claude Haiku
with vision and ask it to pick the best one.

Invoked selectively — expected to trigger for ~5-10% of houses, keeping
cloud costs low.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

import config
from stages.scorer import ScoredFrame


def _encode_image(path: Path, max_bytes: int = config.CLOUD_MAX_IMAGE_BYTES) -> tuple[str, str]:
    """
    Return (base64_data, media_type) for the image at path.
    Resizes if the file exceeds max_bytes to stay within API limits.
    """
    img = Image.open(path)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    if len(data) > max_bytes:
        # Halve dimensions until within budget
        while len(data) > max_bytes and min(img.size) > 200:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()

    return base64.standard_b64encode(data).decode("utf-8"), "image/jpeg"


def pick_best_frame(candidates: list[ScoredFrame]) -> int:
    """
    Ask Claude Haiku to select the best frame from up to 3 candidates.

    Returns the 0-based index into `candidates` of the chosen frame,
    or 0 (local top pick) if the API call fails or returns 'none'.
    """
    import anthropic
    client = anthropic.Anthropic()

    image_blocks = []
    for i, sf in enumerate(candidates):
        b64, media_type = _encode_image(sf.frame_path)
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
        image_blocks.append({
            "type": "text",
            "text": f"Image {i + 1}",
        })

    prompt_text = (
        "You are reviewing street-level photos for HOA compliance documentation. "
        "Each image below is a candidate snapshot of the same residential property.\n\n"
        "Select the single image that best shows:\n"
        "  • The complete front facade of the house (not cropped, not partially hidden)\n"
        "  • A straight-on or near-straight-on angle (minimal perspective distortion)\n"
        "  • No cars, trucks, people, or tree branches blocking the structure\n"
        "  • Sharp focus and good exposure\n\n"
        "Reply with ONLY the number of the best image (1, 2, or 3). "
        "If all images are poor quality, reply with 'none'."
    )

    content = image_blocks + [{"type": "text", "text": prompt_text}]

    try:
        response = client.messages.create(
            model=config.CLOUD_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": content}],
        )
        reply = response.content[0].text.strip().lower()
        if reply in ("1", "2", "3"):
            idx = int(reply) - 1
            return idx if idx < len(candidates) else 0
    except Exception:
        pass

    return 0  # fall back to local top-score pick
