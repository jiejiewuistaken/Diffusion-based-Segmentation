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
from ablation_pred.models.unet3d import UNet3D
from ablation_pred.utils.losses import DiceBCELoss
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
	train_ds = AblationDataset(
		csv_path=data_cfg["train_csv"],
		target_spacing_zyx=tuple(data_cfg["target_spacing"]),
		patch_size_zyx=tuple(data_cfg["patch_size"]),
		augment=bool(data_cfg.get("augment", True)),
		include_probe=True,
	)
	val_ds = AblationDataset(
		csv_path=data_cfg["val_csv"],
		target_spacing_zyx=tuple(data_cfg["target_spacing"]),
		patch_size_zyx=tuple(data_cfg["patch_size"]),
		augment=False,
		include_probe=True,
	)
	train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

	model = UNet3D(
		in_channels=cfg["model"]["in_channels"],
		out_channels=cfg["model"]["out_channels"],
		base_channels=cfg["model"]["base_channels"],
		depth=cfg["model"]["depth"],
	).to(device)
	loss_fn = DiceBCELoss(
		dice_weight=cfg["loss"]["dice_weight"],
		bce_weight=cfg["loss"]["bce_weight"],
	)
	opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
	scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)))
	os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

	best_val = float("inf")
	for epoch in range(cfg["train"]["max_epochs"]):
		model.train()
		loss_meter = AverageMeter()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
		for batch in pbar:
			cond = batch["cond"].to(device)
			target = batch["target"].to(device)
			opt.zero_grad(set_to_none=True)
			with autocast(enabled=bool(cfg["train"].get("mixed_precision", True))):
				logits = model(cond)
				loss = loss_fn(logits, target)
			scaler.scale(loss).backward()
			scaler.step(opt)
			scaler.update()
			loss_meter.update(float(loss.detach().cpu()))
			pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

		# simple val loss
		model.eval()
		val_loss = AverageMeter()
		with torch.no_grad():
			for batch in val_loader:
				cond = batch["cond"].to(device)
				target = batch["target"].to(device)
				logits = model(cond)
				l = loss_fn(logits, target)
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
