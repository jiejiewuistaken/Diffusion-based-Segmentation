"""
Jagged 64x64 ellipse and fixed-area random blob overlay.

This module provides a function to draw two jagged (pixelated) boundaries on top of
an input 2D image:
- An ellipse boundary centered at a given location with specified semi-axes
- A contiguous random blob boundary centered at a given location with specified area

Both boundaries are rasterized on a coarse grid (default 64x64) to produce a
jagged, aliased appearance, and then upsampled with nearest-neighbor to the
original image size and overlaid as colored outlines.

Inputs are image (numpy array or path), ellipse center and axes in pixels, blob
center and target area (in pixels of the original image).

Example
-------
```python
import imageio.v2 as iio
import numpy as np
from jagged_shapes import draw_jagged_shapes

img = np.zeros((480, 640, 3), dtype=np.uint8) + 255
outlined, ell_mask, blob_mask = draw_jagged_shapes(
    image=img,
    ellipse_center=(320, 240),
    ellipse_axes=(150, 90),
    blob_center=(200, 150),
    blob_area=20000,
    grid_size=64,
    ellipse_color=(255, 0, 0),  # red
    blob_color=(0, 255, 0),     # green
    seed=42,
)
# iio.imwrite('outlined.png', outlined)
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import os
import random


Color = Tuple[int, int, int]
ArrayLikeImage = Union[np.ndarray, str]


@dataclass
class ShapeParams:
    ellipse_center: Tuple[float, float]
    ellipse_axes: Tuple[float, float]
    blob_center: Tuple[float, float]
    blob_area: float


def _ensure_image_array(image: ArrayLikeImage) -> np.ndarray:
    """Return image as HxWx3 uint8 numpy array; raises on invalid inputs."""
    if isinstance(image, str):
        # Lazy import to avoid hard dependency if user provides array
        try:
            import imageio.v2 as iio  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Reading image from path requires imageio. Install with `pip install imageio`."
            ) from exc
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        arr = iio.imread(image)
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        # Grayscale -> convert to 3-channel
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Image must be HxWx3 or HxW; got shape {}".format(arr.shape))
    if arr.dtype != np.uint8:
        # Clip and convert
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _coarse_grid_centers(height: int, width: int, grid_h: int, grid_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return meshgrid (Yc, Xc) of coarse cell centers expressed in full-res coordinates.

    Yc, Xc have shape (grid_h, grid_w) giving the full-res y/x coordinate of each
    coarse cell center. Full-res coordinate origin is at the top-left pixel center (0.5, 0.5).
    """
    cell_h = height / grid_h
    cell_w = width / grid_w
    # Coarse centers in full-res coordinates
    yc = (np.arange(grid_h) + 0.5) * cell_h
    xc = (np.arange(grid_w) + 0.5) * cell_w
    Yc, Xc = np.meshgrid(yc, xc, indexing="ij")
    return Yc, Xc


def _map_full_to_coarse(y: float, x: float, height: int, width: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
    """Map a full-res coordinate (x, y) in pixels to a coarse grid index (iy, ix)."""
    iy = int(np.floor(y * grid_h / height))
    ix = int(np.floor(x * grid_w / width))
    iy = int(np.clip(iy, 0, grid_h - 1))
    ix = int(np.clip(ix, 0, grid_w - 1))
    return iy, ix


def _erode_binary_3x3(mask: np.ndarray) -> np.ndarray:
    """Binary erosion with a 3x3 all-ones structuring element using logical AND of shifts."""
    assert mask.ndim == 2 and mask.dtype == bool
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    # Accumulate AND across all 3x3 shifts
    eroded = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return eroded


def _boundary_from_inside(mask_inside: np.ndarray) -> np.ndarray:
    """Return 1-pixel-thick boundary of a binary inside-mask via morphological gradient."""
    eroded = _erode_binary_3x3(mask_inside)
    boundary = mask_inside & (~eroded)
    return boundary


def _nearest_upsample_bool(coarse: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor upsample of a boolean coarse mask to (height, width) without loops."""
    assert coarse.ndim == 2 and coarse.dtype == bool
    grid_h, grid_w = coarse.shape
    # Map full-res indices to coarse indices by flooring
    ys = (np.arange(height) * grid_h) // height
    xs = (np.arange(width) * grid_w) // width
    full = coarse[ys[:, None], xs[None, :]]
    return full


def _ellipse_inside_mask_coarse(height: int, width: int,
                                center: Tuple[float, float],
                                axes: Tuple[float, float],
                                grid_size: int) -> np.ndarray:
    """Compute coarse inside-mask for an ellipse defined in full-res pixel coordinates.

    Returns a boolean array of shape (grid_size, grid_size).
    """
    cy, cx = float(center[1]), float(center[0])  # note: center is (x, y)
    a, b = float(axes[0]), float(axes[1])
    grid_h = grid_w = int(grid_size)
    Yc, Xc = _coarse_grid_centers(height, width, grid_h, grid_w)
    # Ellipse equation: ((x - cx)/a)^2 + ((y - cy)/b)^2 <= 1
    # Clamp axes to avoid division by zero
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    val = ((Xc - cx) / a) ** 2 + ((Yc - cy) / b) ** 2
    inside = val <= 1.0
    return inside


def _random_blob_inside_mask_coarse(height: int, width: int,
                                    center: Tuple[float, float],
                                    target_area_pixels: float,
                                    grid_size: int,
                                    seed: Optional[int]) -> np.ndarray:
    """Grow a contiguous random region on a coarse grid to approximate target area.

    The growth is 4-connected and starts from the coarse cell containing the given center.
    """
    rng = random.Random(seed)
    grid_h = grid_w = int(grid_size)

    # Convert target area in full-res pixels to coarse pixel count
    total_full = float(height * width)
    total_coarse = float(grid_h * grid_w)
    target_coarse = int(round(max(1.0, min(total_coarse, target_area_pixels * total_coarse / total_full))))

    # Map center to coarse index
    iy, ix = _map_full_to_coarse(y=float(center[1]), x=float(center[0]),
                                 height=height, width=width, grid_h=grid_h, grid_w=grid_w)

    inside = np.zeros((grid_h, grid_w), dtype=bool)
    region: set[Tuple[int, int]] = set()
    frontier: set[Tuple[int, int]] = set()

    def add_to_frontier(yx: Tuple[int, int]) -> None:
        y, x = yx
        if 0 <= y < grid_h and 0 <= x < grid_w and (y, x) not in region:
            frontier.add((y, x))

    # Seed with center cell
    region.add((iy, ix))
    inside[iy, ix] = True
    # Initialize frontier with 4-neighbors
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        add_to_frontier((iy + dy, ix + dx))

    # Grow until target size
    while len(region) < target_coarse:
        if not frontier:
            # If frontier exhausted, add a random new frontier from any neighbor of current region
            # This should rarely happen, but ensures progress.
            for (ry, rx) in list(region):
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    add_to_frontier((ry + dy, rx + dx))
            if not frontier:
                # Cannot grow further (already fills grid)
                break
        # Choose a random frontier cell to add
        fy, fx = rng.choice(tuple(frontier))
        frontier.remove((fy, fx))
        # Only add if 4-adjacent to current region to keep connectivity
        is_adjacent = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = fy + dy, fx + dx
            if (ny, nx) in region:
                is_adjacent = True
                break
        if not is_adjacent:
            # If somehow not adjacent (shouldn't happen due to how we build frontier), skip
            continue
        region.add((fy, fx))
        inside[fy, fx] = True
        # Update frontier with neighbors of the newly added cell
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            add_to_frontier((fy + dy, fx + dx))

    return inside


def _overlay_boundaries(
    image: np.ndarray,
    ellipse_boundary: np.ndarray,
    blob_boundary: np.ndarray,
    ellipse_color: Color,
    blob_color: Color,
) -> np.ndarray:
    """Overlay two boolean boundary masks on an image with given colors."""
    out = image.copy()
    assert ellipse_boundary.shape[:2] == image.shape[:2]
    assert blob_boundary.shape[:2] == image.shape[:2]

    ell_mask = ellipse_boundary.astype(bool)
    blob_mask = blob_boundary.astype(bool)
    overlap = ell_mask & blob_mask

    # Apply colors
    if np.any(ell_mask):
        out[ell_mask] = ellipse_color
    if np.any(blob_mask):
        out[blob_mask] = blob_color
    if np.any(overlap):
        # For overlapping pixels, mix colors 50-50
        mixed = tuple(int(0.5 * ec + 0.5 * bc) for ec, bc in zip(ellipse_color, blob_color))
        out[overlap] = mixed

    return out


def draw_jagged_shapes(
    image: ArrayLikeImage,
    ellipse_center: Tuple[float, float],  # (x, y) in pixels
    ellipse_axes: Tuple[float, float],    # (a, b) in pixels (semi-axes)
    blob_center: Tuple[float, float],     # (x, y) in pixels
    blob_area: float,                     # area in full-res pixels
    grid_size: int = 64,
    ellipse_color: Color = (255, 0, 0),
    blob_color: Color = (0, 255, 0),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a jagged ellipse boundary and a contiguous random blob boundary.

    Parameters
    ----------
    image : np.ndarray or str
        Input image. If array, must be HxW or HxWx3 in uint8 or convertible.
        If path, image will be read via imageio.
    ellipse_center : (x, y)
        Ellipse center in pixel coordinates of the full-resolution image.
    ellipse_axes : (a, b)
        Ellipse semi-axes in pixels (horizontal a, vertical b).
    blob_center : (x, y)
        Random blob seed center in pixel coordinates.
    blob_area : float
        Target blob area in pixels of the full-resolution image.
    grid_size : int
        Coarse rasterization size; 64 yields 64x64 jagged boundaries.
    ellipse_color : (R, G, B)
        Color to use for the ellipse boundary.
    blob_color : (R, G, B)
        Color to use for the blob boundary.
    seed : int or None
        Random seed for reproducible blob shapes.

    Returns
    -------
    outlined_image : np.ndarray (HxWx3, uint8)
        The input image with both boundaries overlaid.
    ellipse_boundary_mask : np.ndarray (H x W, bool)
        Boolean mask of the ellipse boundary in full resolution.
    blob_boundary_mask : np.ndarray (H x W, bool)
        Boolean mask of the blob boundary in full resolution.
    """
    img = _ensure_image_array(image)
    height, width = img.shape[:2]
    grid_h = grid_w = int(grid_size)

    # Ellipse inside on coarse grid
    ell_inside_coarse = _ellipse_inside_mask_coarse(
        height=height, width=width, center=ellipse_center, axes=ellipse_axes, grid_size=grid_size
    )
    ell_boundary_coarse = _boundary_from_inside(ell_inside_coarse)

    # Random blob inside on coarse grid
    blob_inside_coarse = _random_blob_inside_mask_coarse(
        height=height, width=width, center=blob_center,
        target_area_pixels=float(blob_area), grid_size=grid_size, seed=seed
    )
    blob_boundary_coarse = _boundary_from_inside(blob_inside_coarse)

    # Upsample boundaries to full resolution
    ell_boundary_full = _nearest_upsample_bool(ell_boundary_coarse, height=height, width=width)
    blob_boundary_full = _nearest_upsample_bool(blob_boundary_coarse, height=height, width=width)

    # Overlay on image
    outlined = _overlay_boundaries(
        image=img,
        ellipse_boundary=ell_boundary_full,
        blob_boundary=blob_boundary_full,
        ellipse_color=ellipse_color,
        blob_color=blob_color,
    )

    return outlined, ell_boundary_full, blob_boundary_full


__all__ = [
    "ShapeParams",
    "draw_jagged_shapes",
]
