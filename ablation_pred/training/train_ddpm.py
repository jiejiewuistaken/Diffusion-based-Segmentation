import os
import argparse
import yaml
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ablation_pred.data.dataset import AblationDataset
from ablation_pred.models.ddpm3d import ConditionalUNetDDPM3D, Diffusion
from ablation_pred.utils.train_utils import seed_everything, get_device, save_checkpoint, AverageMeter


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	with open(args.config, "r") as f:
		cfg: Dict = yaml.safe_load(f)
	seed_everything(cfg.get("seed", 42))
	device = get_device(cfg.get("device", "cuda"))

	data_cfg = cfg["data"]
	cond_scalar_keys = cfg.get("conditioning", {}).get("scalars", [])
	train_ds = AblationDataset(
		csv_path=data_cfg["train_csv"],
		target_spacing_zyx=tuple(data_cfg["target_spacing"]),
		patch_size_zyx=tuple(data_cfg["patch_size"]),
		augment=bool(data_cfg.get("augment", True)),
		include_probe=True,
		include_scalars=cond_scalar_keys,
	)
	val_ds = AblationDataset(
		csv_path=data_cfg["val_csv"],
		target_spacing_zyx=tuple(data_cfg["target_spacing"]),
		patch_size_zyx=tuple(data_cfg["patch_size"]),
		augment=False,
		include_probe=True,
		include_scalars=cond_scalar_keys,
	)
	train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

	model = ConditionalUNetDDPM3D(
		in_channels=cfg["model"]["in_channels"],
		cond_channels=cfg["model"]["cond_channels"],
		base_channels=cfg["model"]["base_channels"],
		depth=cfg["model"]["depth"],
		scalar_cond_dim=len(cond_scalar_keys),
	).to(device)
	diff = Diffusion(
		model=model,
		num_timesteps=cfg["diffusion"]["num_timesteps"],
		beta_schedule=cfg["diffusion"]["beta_schedule"],
		parameterization=cfg["diffusion"]["parameterization"],
	).to(device)

	opt = AdamW(diff.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
	scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)))
	os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

	best_val = float("inf")
	for epoch in range(cfg["train"]["max_epochs"]):
		diff.train()
		loss_meter = AverageMeter()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
		for batch in pbar:
			cond = batch["cond"].to(device)
			scalars = batch["scalars"].to(device) if len(cond_scalar_keys) > 0 else None
			x0 = batch["target"].to(device) * 2.0 - 1.0  # map mask {0,1} to {-1,1}
			opt.zero_grad(set_to_none=True)
			t = torch.randint(0, cfg["diffusion"]["num_timesteps"], (x0.shape[0],), device=device)
			with autocast(enabled=bool(cfg["train"].get("mixed_precision", True))):
				loss = diff.p_losses(x0, cond, t, scalars)
			scaler.scale(loss).backward()
			scaler.step(opt)
			scaler.update()
			loss_meter.update(float(loss.detach().cpu()))
			pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

		# simple val N-step sampling loss proxy (optional): use fixed t
		diff.eval()
		val_loss = AverageMeter()
		with torch.no_grad():
			for batch in val_loader:
				cond = batch["cond"].to(device)
				scalars = batch["scalars"].to(device) if len(cond_scalar_keys) > 0 else None
				x0 = batch["target"].to(device) * 2.0 - 1.0
				t = torch.full((1,), cfg["diffusion"]["num_timesteps"] - 1, device=device, dtype=torch.long)
				l = diff.p_losses(x0, cond, t, scalars)
				val_loss.update(float(l.detach().cpu()))

		state = {
			"epoch": epoch,
			"model": model.state_dict(),
			"opt": opt.state_dict(),
		}
		save_checkpoint(state, os.path.join(cfg["train"]["save_dir"], "last.ckpt"))
		if val_loss.avg < best_val:
			best_val = val_loss.avg
			save_checkpoint(state, os.path.join(cfg["train"]["save_dir"], "best.ckpt"))
		print(f"Epoch {epoch}: train {loss_meter.avg:.4f} val {val_loss.avg:.4f}")


if __name__ == "__main__":
	main()
