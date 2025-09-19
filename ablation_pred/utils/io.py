import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


def read_csv_records(csv_path: str) -> List[Dict[str, str]]:
	csv_rows: List[Dict[str, str]] = []
	with open(csv_path, "r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			csv_rows.append({k: (v if v is not None else "") for k, v in row.items()})
	return csv_rows


def load_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
	image = sitk.ReadImage(path)
	array = sitk.GetArrayFromImage(image).astype(np.float32)  # z, y, x
	spacing = tuple(reversed(image.GetSpacing()))  # sitk spacing is x,y,z
	return array, spacing  # spacing as (z, y, x)


def _to_sitk_image(array_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> sitk.Image:
	img = sitk.GetImageFromArray(array_zyx)
	# convert (z,y,x) spacing to (x,y,z)
	img.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
	return img


def resample_to_spacing(
	array_zyx: np.ndarray,
	spacing_zyx: Tuple[float, float, float],
	target_spacing_zyx: Tuple[float, float, float],
	is_label: bool,
) -> np.ndarray:
	img = _to_sitk_image(array_zyx, spacing_zyx)
	interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
	size = np.array(array_zyx.shape, dtype=np.int64)
	new_size = np.maximum(
		1,
		(np.round(size * np.array(spacing_zyx) / np.array(target_spacing_zyx))).astype(np.int64),
	)
	resampled = sitk.Resample(
		img,
		size=tuple(int(v) for v in new_size[::-1]),  # sitk uses x,y,z
		transform=sitk.Transform(),
		interpolator=interp,
		outputOrigin=img.GetOrigin(),
		outputSpacing=(target_spacing_zyx[2], target_spacing_zyx[1], target_spacing_zyx[0]),
		outputDirection=img.GetDirection(),
	)
	return sitk.GetArrayFromImage(resampled).astype(np.float32)


def normalize_ct_hu(ct_zyx: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
	ct = np.clip(ct_zyx, hu_min, hu_max)
	ct = (ct - hu_min) / (hu_max - hu_min)
	ct = ct * 2.0 - 1.0
	return ct.astype(np.float32)


def maybe_load_mask(path: Optional[str]) -> Optional[Tuple[np.ndarray, Tuple[float, float, float]]]:
	if path is None or path == "":
		return None
	return load_nifti(path)
