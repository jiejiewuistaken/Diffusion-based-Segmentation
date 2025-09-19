import os
import random
from typing import Dict, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
	if device_str == "cuda" and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def save_checkpoint(state: Dict, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
	return torch.load(path, map_location=map_location)


class AverageMeter:
	def __init__(self):
		self.reset()

	def reset(self):
		self.sum = 0.0
		self.count = 0

	def update(self, val: float, n: int = 1):
		self.sum += float(val) * n
		self.count += n

	@property
	def avg(self) -> float:
		return self.sum / max(1, self.count)
