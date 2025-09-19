from typing import List

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
		nn.InstanceNorm3d(out_ch, affine=True),
		nn.GELU(),
		nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
		nn.InstanceNorm3d(out_ch, affine=True),
		nn.GELU(),
	)


class UNet3D(nn.Module):
	def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
		super().__init__()
		self.depth = depth
		chs: List[int] = [base_channels * (2 ** i) for i in range(depth)]
		self.downs = nn.ModuleList()
		self.pools = nn.ModuleList()
		prev = in_channels
		for c in chs:
			self.downs.append(conv_block(prev, c))
			self.pools.append(nn.MaxPool3d(kernel_size=2))
			prev = c
		self.bottleneck = conv_block(prev, prev * 2)
		prev = prev * 2
		self.ups = nn.ModuleList()
		self.up_convs = nn.ModuleList()
		for c in reversed(chs):
			self.ups.append(nn.ConvTranspose3d(prev, c, kernel_size=2, stride=2))
			self.up_convs.append(conv_block(prev, c))
			prev = c
		self.head = nn.Conv3d(prev, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		skips: List[torch.Tensor] = []
		out = x
		for i in range(self.depth):
			out = self.downs[i](out)
			skips.append(out)
			out = self.pools[i](out)
		out = self.bottleneck(out)
		for i in range(self.depth - 1, -1, -1):
			out = self.ups[self.depth - 1 - i](out)
			out = torch.cat([out, skips[i]], dim=1)
			out = self.up_convs[self.depth - 1 - i](out)
		logits = self.head(out)
		return logits
