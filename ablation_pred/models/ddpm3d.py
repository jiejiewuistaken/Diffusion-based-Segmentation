from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
	# timesteps: (N,)
	half = dim // 2
	freqs = torch.exp(-torch.arange(half, device=timesteps.device, dtype=torch.float32) * (torch.log(torch.tensor(max_period, dtype=torch.float32, device=timesteps.device)) / half))
	args = timesteps.float()[:, None] * freqs[None]
	emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
	return emb


class ResBlock(nn.Module):
	def __init__(self, in_ch: int, out_ch: int, time_dim: int):
		super().__init__()
		self.norm1 = nn.InstanceNorm3d(in_ch, affine=True)
		self.act = nn.GELU()
		self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
		self.time_fc = nn.Linear(time_dim, out_ch)
		self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
		self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
		self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

	def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		out = self.conv1(self.act(self.norm1(x)))
		out = out + rearrange(self.time_fc(t_emb), "n c -> n c 1 1 1")
		out = self.conv2(self.act(self.norm2(out)))
		return out + self.skip(x)


class Down(nn.Module):
	def __init__(self, in_ch: int, out_ch: int, time_dim: int):
		super().__init__()
		self.block = ResBlock(in_ch, out_ch, time_dim)
		self.pool = nn.AvgPool3d(2)

	def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
		x = self.block(x, t_emb)
		return self.pool(x), x


class Up(nn.Module):
	def __init__(self, in_ch: int, out_ch: int, time_dim: int):
		super().__init__()
		self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
		self.block = ResBlock(out_ch * 2, out_ch, time_dim)

	def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
		x = self.up(x)
		x = torch.cat([x, skip], dim=1)
		return self.block(x, t_emb)


class ConditionalUNetDDPM3D(nn.Module):
	def __init__(self, in_channels: int, cond_channels: int, base_channels: int = 64, depth: int = 4, time_dim: int = 256, scalar_cond_dim: int = 0):
		super().__init__()
		self.time_mlp = nn.Sequential(
			nn.Linear(time_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
		)
		self.scalar_mlp = None
		if scalar_cond_dim > 0:
			self.scalar_mlp = nn.Sequential(
				nn.Linear(scalar_cond_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
			)
		self.in_conv = nn.Conv3d(in_channels + cond_channels, base_channels, 3, padding=1)
		chs = [base_channels * (2 ** i) for i in range(depth)]
		self.downs = nn.ModuleList()
		self.skips_proj = nn.ModuleList()
		ch = base_channels
		for c in chs:
			self.downs.append(Down(ch, c, time_dim))
			self.skips_proj.append(nn.Identity())
			ch = c
		self.mid1 = ResBlock(ch, ch, time_dim)
		self.mid2 = ResBlock(ch, ch, time_dim)
		self.ups = nn.ModuleList()
		for c in reversed(chs):
			self.ups.append(Up(ch, c, time_dim))
			ch = c
		self.out_norm = nn.InstanceNorm3d(ch, affine=True)
		self.out_act = nn.GELU()
		self.out_conv = nn.Conv3d(ch, in_channels, 3, padding=1)

	def forward(self, x_t: torch.Tensor, cond_img: torch.Tensor, t: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
		# x_t: (N,1,D,H,W), cond_img: (N,Cc,D,H,W)
		t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
		t_emb = self.time_mlp(t_emb)
		if self.scalar_mlp is not None and scalar_cond is not None:
			t_emb = t_emb + self.scalar_mlp(scalar_cond)
		out = torch.cat([x_t, cond_img], dim=1)
		out = self.in_conv(out)
		skips = []
		for down in self.downs:
			out, skip = down(out, t_emb)
			skips.append(skip)
		out = self.mid2(self.mid1(out, t_emb), t_emb)
		for up in self.ups:
			skip = skips.pop()
			out = up(out, skip, t_emb)
		out = self.out_conv(self.out_act(self.out_norm(out)))
		return out


class Diffusion(nn.Module):
	def __init__(self, model: ConditionalUNetDDPM3D, num_timesteps: int = 1000, beta_schedule: str = "linear", parameterization: str = "eps"):
		super().__init__()
		self.model = model
		self.parameterization = parameterization
		if beta_schedule == "linear":
			betas = torch.linspace(1e-4, 0.02, num_timesteps)
		else:
			raise ValueError("Unsupported beta schedule")
		self.register_buffer("betas", betas)
		alphas = 1.0 - betas
		alphas_cum = torch.cumprod(alphas, dim=0)
		self.register_buffer("alphas_cumprod", alphas_cum)
		self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cum))
		self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cum))
		self.register_buffer("one_over_sqrt_alpha", torch.sqrt(1.0 / alphas))

	def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
		if noise is None:
			noise = torch.randn_like(x0)
		return self.sqrt_alphas_cumprod[t][:, None, None, None, None] * x0 + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None] * noise

	def p_losses(self, x0: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, scalar_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
		noise = torch.randn_like(x0)
		x_t = self.q_sample(x0, t, noise)
		pred = self.model(x_t, cond, t, scalar_cond)
		if self.parameterization == "eps":
			return torch.mean((pred - noise) ** 2)
		else:
			raise ValueError("Unsupported parameterization")

	@torch.no_grad()
	def ddim_sample(self, shape, cond: torch.Tensor, scalar_cond: Optional[torch.Tensor], steps: int = 100, eta: float = 0.0, device: Optional[torch.device] = None) -> torch.Tensor:
		device = device or cond.device
		b = shape[0]
		x = torch.randn(shape, device=device)
		T = self.betas.shape[0]
		ids = torch.linspace(T - 1, 0, steps, device=device).long()
		alphas = 1.0 - self.betas
		alphas_cum = torch.cumprod(alphas, dim=0)
		for i in range(steps):
			t = ids[i].expand(b)
			alpha_t = alphas[t][:, None, None, None, None]
			alpha_bar_t = alphas_cum[t][:, None, None, None, None]
			pred_eps = self.model(x, cond, t, scalar_cond)
			x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
			if i == steps - 1:
				x = x0_pred
				break
			alpha_bar_next = alphas_cum[ids[i + 1]][:, None, None, None, None]
			sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
			dir_xt = torch.sqrt(1 - alpha_bar_next) * pred_eps
			x = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt + sigma * torch.randn_like(x)
		return x
