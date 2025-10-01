"""Add Gaussian noise to a 64x64 CT 2D image over multiple timesteps and save a tiled PNG.

This script reads a grayscale image, progressively adds independent Gaussian noise at each
time step, and saves the resulting noisy images tiled side-by-side into a single PNG.

Usage example:
    python noise_sequence.py --input /path/to/ct.png --output /path/to/noisy_steps.png \
        --steps 10 --sigma 0.1 --seed 42
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
from PIL import Image


def load_grayscale_image_as_float(path: str, target_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load an image as grayscale float32 in [0, 1]. Optionally resize to target_size.

    Args:
        path: Path to the input image.
        target_size: Desired (width, height). Defaults to (64, 64).

    Returns:
        NumPy array of shape (H, W) with dtype float32, values in [0, 1].
    """
    img = Image.open(path).convert("L")  # grayscale
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    # Normalize to [0, 1]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def add_gaussian_noise_sequence(
    image01: np.ndarray,
    steps: int = 10,
    sigma: float = 0.1,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Progressively add Gaussian noise for a number of steps.

    Each step adds independent N(0, sigma^2) noise to the previous result.

    Args:
        image01: Input image in [0, 1], shape (H, W).
        steps: Number of time steps.
        sigma: Standard deviation of the Gaussian noise (on [0, 1] scale).
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of images (length = steps), each in [0, 1], shape (H, W).
    """
    if image01.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    if steps <= 0:
        raise ValueError("'steps' must be a positive integer.")
    if sigma < 0:
        raise ValueError("'sigma' must be non-negative.")

    rng = np.random.default_rng(seed)
    current = image01.astype(np.float32, copy=True)
    results: List[np.ndarray] = []

    for _ in range(steps):
        noise = rng.normal(loc=0.0, scale=sigma, size=current.shape).astype(np.float32)
        current = np.clip(current + noise, 0.0, 1.0)
        results.append(current.copy())

    return results


def tile_side_by_side_gray(images01: List[np.ndarray]) -> Image.Image:
    """Tile grayscale images side-by-side and return a PIL Image in mode 'L'.

    Args:
        images01: List of images in [0, 1], each with shape (H, W).

    Returns:
        A PIL Image where the images are concatenated along width, 8-bit grayscale.
    """
    if not images01:
        raise ValueError("No images provided to tile.")

    heights = [img.shape[0] for img in images01]
    widths = [img.shape[1] for img in images01]
    if len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError("All images must have the same shape.")

    uint8_images = [(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8) for img in images01]
    concatenated = np.concatenate(uint8_images, axis=1)
    return Image.fromarray(concatenated, mode="L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Gaussian noise over timesteps and tile results.")
    parser.add_argument("--input", required=True, help="Path to the 64x64 grayscale CT image (or any size).")
    parser.add_argument("--output", default="noisy_steps.png", help="Path to save the tiled PNG.")
    parser.add_argument("--steps", type=int, default=10, help="Number of time steps (default: 10).")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Gaussian noise standard deviation on [0,1] scale (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image01 = load_grayscale_image_as_float(args.input, target_size=(64, 64))
    noisy_images = add_gaussian_noise_sequence(
        image01=image01, steps=args.steps, sigma=args.sigma, seed=args.seed
    )
    tiled = tile_side_by_side_gray(noisy_images)
    tiled.save(args.output)


if __name__ == "__main__":
    main()

