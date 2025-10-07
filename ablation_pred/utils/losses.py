import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
	def __init__(self, eps: float = 1e-6):
		super().__init__()
		self.eps = eps

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		# logits: (N,1,D,H,W), target: (N,1,D,H,W)
		probs = torch.sigmoid(logits)
		inter = (probs * target).sum(dim=(2, 3, 4))
		sums = (probs + target).sum(dim=(2, 3, 4))
		dice = (2.0 * inter + self.eps) / (sums + self.eps)
		return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
	def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
		super().__init__()
		self.dice = SoftDiceLoss()
		self.bce = nn.BCEWithLogitsLoss()
		self.dw = dice_weight
		self.bw = bce_weight

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return self.dw * self.dice(logits, target) + self.bw * self.bce(logits, target)
