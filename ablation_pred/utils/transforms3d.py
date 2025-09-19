from typing import Tuple

import numpy as np


def center_crop_or_pad(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
	assert array.ndim == 3
	out = np.zeros(target_shape, dtype=array.dtype)
	input_shape = np.array(array.shape)
	target_shape_np = np.array(target_shape)
	start_src = np.maximum(0, (input_shape - target_shape_np) // 2)
	end_src = start_src + np.minimum(input_shape, target_shape_np)
	start_dst = np.maximum(0, (target_shape_np - input_shape) // 2)
	end_dst = start_dst + (end_src - start_src)
	out[start_dst[0]:end_dst[0], start_dst[1]:end_dst[1], start_dst[2]:end_dst[2]] = array[
		start_src[0]:end_src[0], start_src[1]:end_src[1], start_src[2]:end_src[2]
	]
	return out


def extract_patch_centered(array: np.ndarray, center_zyx: Tuple[int, int, int], patch_shape: Tuple[int, int, int]) -> np.ndarray:
	z, y, x = center_zyx
	hz, hy, hx = patch_shape[0] // 2, patch_shape[1] // 2, patch_shape[2] // 2
	z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hx)
	z1, y1, x1 = min(array.shape[0], z + hz), min(array.shape[1], y + hy), min(array.shape[2], x + hx)
	patch = np.zeros(patch_shape, dtype=array.dtype)
	pz0, py0, px0 = hz - (z - z0), hy - (y - y0), hx - (x - x0)
	pz1, py1, px1 = pz0 + (z1 - z0), py0 + (y1 - y0), px0 + (x1 - x0)
	patch[pz0:pz1, py0:py1, px0:px1] = array[z0:z1, y0:y1, x0:x1]
	return patch


def random_flip_3d(array: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
	out = array
	for axis in range(3):
		if rng.rand() < 0.5:
			out = np.flip(out, axis=axis)
	return out.copy()
