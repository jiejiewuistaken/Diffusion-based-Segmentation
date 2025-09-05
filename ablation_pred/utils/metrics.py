from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi


def dice_coefficient(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
	a_bin = (a > 0.5).astype(np.float32)
	b_bin = (b > 0.5).astype(np.float32)
	inter = float((a_bin * b_bin).sum())
	return (2.0 * inter + eps) / (float(a_bin.sum() + b_bin.sum()) + eps)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
	# surfaces via binary erosion difference
	struct = ndi.generate_binary_structure(3, 1)
	a_surf = a.astype(bool) ^ ndi.binary_erosion(a.astype(bool), structure=struct, iterations=1, border_value=0)
	b_surf = b.astype(bool) ^ ndi.binary_erosion(b.astype(bool), structure=struct, iterations=1, border_value=0)
	if not a_surf.any():
		return np.array([0.0], dtype=np.float32)
	if not b_surf.any():
		return np.array([0.0], dtype=np.float32)
	dist_map = ndi.distance_transform_edt(~b.astype(bool), sampling=spacing)
	return dist_map[a_surf]


def hd95(a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]) -> float:
	a = (a > 0.5)
	b = (b > 0.5)
	d1 = _surface_distances(a, b, spacing)
	d2 = _surface_distances(b, a, spacing)
	if d1.size == 0 and d2.size == 0:
		return 0.0
	return float(np.percentile(np.concatenate([d1, d2]), 95))


def volume_error_ml(a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]) -> float:
	voxel_ml = spacing[0] * spacing[1] * spacing[2]
	va = float((a > 0.5).sum()) * voxel_ml
\tvb = float((b > 0.5).sum()) * voxel_ml
	return abs(va - vb)


def coverage_metrics(
	tumor: np.ndarray,
	ablation: np.ndarray,
	spacing: Tuple[float, float, float],
	margin_mm: float = 5.0,
) -> Dict[str, float]:
	dtumor = ndi.distance_transform_edt(~(tumor > 0.5), sampling=spacing)
	margin_region = (dtumor <= margin_mm)
	covered_tumor = float(((tumor > 0.5) & (ablation > 0.5)).sum())
	tumor_voxels = float((tumor > 0.5).sum() + 1e-6)
	covered_margin = float((margin_region & (ablation > 0.5)).sum())
	margin_voxels = float(margin_region.sum() + 1e-6)
	return {
		"tumor_coverage": covered_tumor / tumor_voxels,
		"margin_coverage": covered_margin / margin_voxels,
	}
