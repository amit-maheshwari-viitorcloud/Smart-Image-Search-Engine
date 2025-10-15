import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_image_from_path(path: str) -> Optional[Image.Image]:
    """Load image from local path or URL"""
    try:
        if path.startswith(("http://", "https://")):
            response = requests.get(path, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        return image
    except (requests.RequestException, FileNotFoundError, UnidentifiedImageError, OSError) as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


def validate_image(image_file) -> bool:
    """Validate uploaded image file"""
    if image_file is None:
        return False
    
    try:
        image = Image.open(image_file)
        image.verify()
        return True
    except Exception:
        return False

