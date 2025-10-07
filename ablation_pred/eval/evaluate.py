import os
import argparse
import yaml
from typing import Dict

import torch
from torch.utils.data import DataLoader
import numpy as np

from ablation_pred.data.dataset import AblationDataset
from ablation_pred.models.unet3d import UNet3D
from ablation_pred.models.ddpm3d import ConditionalUNetDDPM3D, Diffusion
from ablation_pred.utils.train_utils import seed_everything, get_device, load_checkpoint
from ablation_pred.utils.metrics import dice_coefficient, hd95, volume_error_ml, coverage_metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	parser.add_argument("--ckpt", type=str, required=True)
	parser.add_argument("--model", type=str, default="unet", choices=["unet", "ddpm"])
	parser.add_argument("--samples", type=int, default=8)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	with open(args.config, "r") as f:
		cfg: Dict = yaml.safe_load(f)
	seed_everything(cfg.get("seed", 42))
	device = get_device(cfg.get("device", "cuda"))

	data_cfg = cfg["data"]
	test_ds = AblationDataset(
		csv_path=data_cfg["test_csv"],
		target_spacing_zyx=tuple(data_cfg["target_spacing"]),
		patch_size_zyx=tuple(data_cfg["patch_size"]),
		augment=False,
		include_probe=True,
		include_scalars=cfg.get("conditioning", {}).get("scalars", []),
	)
	test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

	if args.model == "unet":
		model = UNet3D(
			in_channels=cfg["model"]["in_channels"],
			out_channels=cfg["model"]["out_channels"],
			base_channels=cfg["model"]["base_channels"],
			depth=cfg["model"]["depth"],
		).to(device)
		state = load_checkpoint(args.ckpt, map_location=str(device))
		model.load_state_dict(state["model"])
		model.eval()
	else:
		model = ConditionalUNetDDPM3D(
			in_channels=cfg["model"]["in_channels"],
			cond_channels=cfg["model"]["cond_channels"],
			base_channels=cfg["model"]["base_channels"],
			depth=cfg["model"]["depth"],
			scalar_cond_dim=len(cfg.get("conditioning", {}).get("scalars", [])),
		).to(device)
		diff = Diffusion(
			model=model,
			num_timesteps=cfg["diffusion"]["num_timesteps"],
			beta_schedule=cfg["diffusion"]["beta_schedule"],
			parameterization=cfg["diffusion"]["parameterization"],
		).to(device)
		state = load_checkpoint(args.ckpt, map_location=str(device))
		model.load_state_dict(state["model"])
		model.eval()

	all_metrics = []
	with torch.no_grad():
		for batch in test_loader:
			cond = batch["cond"].to(device)
			gt = batch["target"].to(device)
			spacing = test_ds.target_spacing
			if args.model == "unet":
				logits = model(cond)
				pred = (torch.sigmoid(logits) > cfg.get("inference", {}).get("threshold", 0.5)).float()
			else:
				scalars = batch["scalars"].to(device) if len(cfg.get("conditioning", {}).get("scalars", [])) > 0 else None
				samples = []
				for _ in range(args.samples):
					x = diff.ddim_sample(
						shape=gt.shape,
						cond=cond,
						scalar_cond=scalars,
						steps=cfg["diffusion"].get("ddim_steps", 100),
						eta=cfg["diffusion"].get("ddim_eta", 0.0),
						device=device,
					)
					samples.append((x + 1.0) / 2.0)  # back to [0,1]
				mean_pred = torch.stack(samples, dim=0).mean(dim=0)
				pred = (mean_pred > 0.5).float()

			pred_np = pred.squeeze().cpu().numpy()
			gt_np = gt.squeeze().cpu().numpy()
			m = {
				"dice": dice_coefficient(pred_np, gt_np),
				"hd95": hd95(pred_np, gt_np, spacing),
				"vol_err_ml": volume_error_ml(pred_np, gt_np, spacing),
			}
			# Optional: tumor/margin coverage if tumor mask available in cond channel 2
			tumor_np = (cond[:, 1:2].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
			cov = coverage_metrics(tumor_np, pred_np, spacing)
			m.update(cov)
			all_metrics.append(m)
			print(m)

	# aggregate
	keys = all_metrics[0].keys()
	avg = {k: float(np.mean([mm[k] for mm in all_metrics])) for k in keys}
	print({"mean": avg})


if __name__ == "__main__":
	main()
