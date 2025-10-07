import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ablation_pred.utils.io import (
	load_nifti,
	maybe_load_mask,
	read_csv_records,
	resample_to_spacing,
	normalize_ct_hu,
)
from ablation_pred.utils.transforms3d import center_crop_or_pad, random_flip_3d


class AblationDataset(Dataset):
	def __init__(
		self,
		csv_path: str,
		target_spacing_zyx: Tuple[float, float, float],
		patch_size_zyx: Tuple[int, int, int],
		augment: bool = False,
		include_probe: bool = True,
		include_scalars: Optional[List[str]] = None,
	):
		super().__init__()
		self.records = read_csv_records(csv_path)
		self.target_spacing = target_spacing_zyx
		self.patch_size = patch_size_zyx
		self.augment = augment
		self.include_probe = include_probe
		self.include_scalars = include_scalars or []

	def __len__(self) -> int:
		return len(self.records)

	def _build_condition(self, rec: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
		ct, ct_spacing = load_nifti(rec["ct_path"])  # z,y,x
		ct = resample_to_spacing(ct, ct_spacing, self.target_spacing, is_label=False)
		ct = normalize_ct_hu(ct)
		tumor, tumor_spacing = load_nifti(rec["tumor_mask_path"])  # z,y,x
		tumor = resample_to_spacing(tumor, tumor_spacing, self.target_spacing, is_label=True)
		probe_array = None
		if self.include_probe:
			probe = maybe_load_mask(rec.get("probe_mask_path"))
			if probe is not None:
				probe_array = resample_to_spacing(probe[0], probe[1], self.target_spacing, is_label=True)
			else:
				probe_array = np.zeros_like(tumor, dtype=np.float32)
		cond = [ct, tumor]
		if probe_array is not None:
			cond.append(probe_array)
		cond = np.stack(cond, axis=0)  # c,z,y,x
		scalars = []
		for key in self.include_scalars:
			val = rec.get(key, "")
			if val == "":
				scalars.append(0.0)
			else:
				scalars.append(float(val))
		return cond.astype(np.float32), np.array(scalars, dtype=np.float32)

	def _load_target(self, rec: Dict[str, str]) -> np.ndarray:
		y, y_spacing = load_nifti(rec["target_mask_path"])  # ablation mask
		y = resample_to_spacing(y, y_spacing, self.target_spacing, is_label=True)
		return y.astype(np.float32)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		rec = self.records[idx]
		cond, scalars = self._build_condition(rec)
		y = self._load_target(rec)

		# center-crop/pad to patch size
		cond_c = np.stack([center_crop_or_pad(c, self.patch_size) for c in cond], axis=0)
		y_c = center_crop_or_pad(y, self.patch_size)

		if self.augment:
			rng = np.random.RandomState(seed=idx)
			cond_c = np.stack([random_flip_3d(c, rng) for c in cond_c], axis=0)
			y_c = random_flip_3d(y_c, rng)

		return {
			"cond": torch.from_numpy(cond_c),  # (C,D,H,W)
			"scalars": torch.from_numpy(scalars),
			"target": torch.from_numpy(y_c[None, ...]),  # (1,D,H,W)
		}
