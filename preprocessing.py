"""
Image preprocessing utilities for Eye Disease Classification.
Provides reusable, robust image validation, conversion, and normalization.
"""

import io
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import IMAGE_SIZE, CHANNELS, INPUT_SHAPE


class PreprocessingError(Exception):
    """Custom exception raised when image preprocessing fails."""
    pass


def validate_image_input(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]) -> None:
    """
    Validate that the input is a recognized type and not empty.
    """
    if image_input is None:
        raise PreprocessingError("Image input cannot be None.")
    
    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise PreprocessingError("Image byte buffer is empty.")
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.is_file() and path.stat().st_size == 0:
            raise PreprocessingError(f"Image file is empty (0 bytes): {path}")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise PreprocessingError("Image numpy array is empty.")


def to_pil_image(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Image.Image:
    """
    Safely load an arbitrary image input into a PIL.Image instance.
    """
    validate_image_input(image_input)

    if isinstance(image_input, Image.Image):
        return image_input.copy()

    if isinstance(image_input, (str, Path)):
        try:
            return Image.open(image_input)
        except UnidentifiedImageError as err:
            raise PreprocessingError(f"Cannot identify image file '{image_input}': corrupt or unsupported format.") from err
        except Exception as err:
            raise PreprocessingError(f"Failed to open image file '{image_input}': {err}") from err

    if isinstance(image_input, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image_input))
        except UnidentifiedImageError as err:
            raise PreprocessingError("Corrupted or invalid image bytes provided.") from err
        except Exception as err:
            raise PreprocessingError(f"Failed to decode image bytes: {err}") from err

    if hasattr(image_input, "read"):
        try:
            image_bytes = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            if not image_bytes:
                raise PreprocessingError("Stream read 0 bytes.")
            return Image.open(io.BytesIO(image_bytes))
        except Exception as err:
            raise PreprocessingError(f"Failed to read image stream: {err}") from err

    if isinstance(image_input, np.ndarray):
        arr = image_input
        # If float in [0, 1], scale to [0, 255] for PIL conversion
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

        # Handle squeezed 2D or 3D
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0], mode="L")
            elif arr.shape[2] == 3:
                return Image.fromarray(arr, mode="RGB")
            elif arr.shape[2] == 4:
                return Image.fromarray(arr, mode="RGBA")
            else:
                raise PreprocessingError(f"Unsupported channel dimension in array: {arr.shape[2]}")
        else:
            raise PreprocessingError(f"Unsupported numpy array dimensions: {arr.ndim}")

    raise PreprocessingError(f"Unsupported image input type: {type(image_input).__name__}")


def preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    target_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Standard preprocessing pipeline applied identically during training, testing, and inference:
    1. Validate input.
    2. Convert to PIL Image.
    3. Convert to RGB (ensures exactly 3 channels even for grayscale or RGBA).
    4. Resize to target_size (default: 28x28).
    5. Convert to np.float32.
    6. Normalize pixel values to [0.0, 1.0].

    Returns:
        np.ndarray of shape (target_size[0], target_size[1], 3) with float32 values in [0.0, 1.0].
    """
    pil_img = to_pil_image(image_input)

    # 1. Convert to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 2. Resize to 28x28
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, resample=Image.Resampling.BILINEAR)

    # 3. Convert to float32 numpy array
    arr = np.asarray(pil_img, dtype=np.float32)

    # 4. Normalize to range [0.0, 1.0]
    arr = arr / 255.0

    # Ensure bounds and shape
    if arr.shape != INPUT_SHAPE:
        raise PreprocessingError(f"Preprocessed shape {arr.shape} does not match expected {INPUT_SHAPE}")

    return arr


def preprocess_image_for_prediction(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    target_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Preprocess a single image and expand its batch dimension for CNN inference.
    Returns:
        np.ndarray of shape (1, 28, 28, 3) with float32 values in [0.0, 1.0].
    """
    preprocessed = preprocess_image(image_input, target_size=target_size)
    return np.expand_dims(preprocessed, axis=0)


def preprocess_batch(images: np.ndarray) -> np.ndarray:
    """
    Preprocess a batch of raw uint8 images (e.g. directly from MedMNIST npz).
    Converts to float32 and scales [0, 255] -> [0.0, 1.0].
    """
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)

    if images.ndim != 4 or images.shape[1:4] != INPUT_SHAPE:
        raise PreprocessingError(
            f"Expected batch shape (N, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}, {INPUT_SHAPE[2]}), got {images.shape}"
        )

    images = images.astype(np.float32)

    # If images are in 0-255 range, normalize
    if images.max() > 1.0:
        images = images / 255.0

    return np.clip(images, 0.0, 1.0)

