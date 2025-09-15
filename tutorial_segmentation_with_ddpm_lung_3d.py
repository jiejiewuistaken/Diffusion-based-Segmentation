# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

#
# # 3D Diffusion Models for Implicit Image Segmentation Ensembles<br>
# <br>
# This tutorial illustrates how to use MONAI for 3D segmentation of volumes using DDPMs, adapted from [1].<br>
# <br>
# [1] - Wolleb et al. "Diffusion Models for Implicit Image Segmentation Ensembles", https://arxiv.org/abs/2112.03145<br>
#

# ## Setup environment

# !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# !python -c "import seaborn" || pip install -q seaborn

#
# ## Setup imports

# +
import os
import tempfile
import time
import logging

import sys
sys.path.append('/home/yw4445/GenerativeModels/')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")
print_config()
# -


config = {
    'model_name': 'Lung3DDDPM-UNet',
    'dataset': 'Task06_Lung',
    'mode': '3d',
    'image_size': (64, 64, 64),  # Changed to 3D dimensions
    'batch_size': 4,  # Reduced batch size due to 3D memory requirements
    'epochs': 6000,
    'unet_channels': (64, 128, 256, 256),  # Adjusted channel sizes for 3D
    'attention_levels': (False, False, False, False),  # Added attention to deeper levels
    'num_res_blocks': 2,
    'lr': 1e-4,
    'notes': "Use Florentin suggested params. pre-cropped. "
}

from datetime import datetime
import os

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ## Setup data directory

# os.environ["MONAI_DATA_DIRECTORY"] = '/tmp/tmpza1ws9y6/'
os.environ["MONAI_DATA_DIRECTORY"] = '/tmp/tmpza1ws9y6/preprocessed_lung'
# os.environ["MONAI_DATA_DIRECTORY"] = '/tmp/tmpdfo8fixc' # for Task06

directory = os.environ.get("MONAI_DATA_DIRECTORY")
# directory = None
root_dir = tempfile.mkdtemp() if directory is None else directory

result_image_dir = f'/home/yw4445/GenerativeModelTutorialOutput/{config["dataset"]}/3D_runs_{timestamp}/'
os.makedirs(result_image_dir, exist_ok=True)
log_file = result_image_dir + 'training.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write to file
        logging.StreamHandler()  # Print to console
    ]
)

logger = logging.getLogger(__name__)

import json

# JSON doesn't support tuples, need to convert to lists
config_json = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}

# Save to JSON file
with open(result_image_dir + "config.json", "w") as f:
    json.dump(config_json, f, indent=4)

#
# ## Set deterministic training for reproducibility

set_determinism(42)

#
# # Preprocessing of the dataset for 3D volume segmentation
# We prepare the data for 3D segmentation by keeping the volumes in 3D format.

# +
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

# Define 3D transforms that maintain the 3D structure
train_transforms = transforms.Compose(
    
#     [
#     # 1. Load the data
#     transforms.LoadImaged(keys=["image", "label"]),
#     transforms.EnsureChannelFirstd(keys=["image", "label"]),
#     transforms.EnsureTyped(keys=["image", "label"]),
    
#     # 2. 3D preprocessing
#     transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#     transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
    
    
#     transforms.RandCropByPosNegLabeld(
#         keys=["image", "label"],  # 需要裁剪的 key
#         label_key="label",        # 以 label 为参考
#         spatial_size=(64, 64, 64), # 目标裁剪尺寸
#         pos=1.0,                   # 100% 采样病灶区域
#         neg=0.0,                   # 不采样负样本（背景）
#         num_samples=1,              # 每个样本裁剪 1 个 patch
#     ),
#     transforms.Lambdad(keys=["image", "label"], func=lambda x: x[0]),
    
#     # 3. Center crop to target 3D dimensions
#     # transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=config['image_size']),
    
#     # 4. Scale intensity
#     transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
# ]
# 可以用的！！！只是说会把label弄的很大
# [
#                 transforms.LoadImaged(keys=["image", "label"]),
#                 transforms.EnsureChannelFirstd(keys=["image", "label"]),
#                 transforms.ScaleIntensityRanged(
#                     keys=["image"],
#                     a_min=-175,
#                     a_max=250,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=True,
#                 ),
#                 # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
#                 transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#                 transforms.Spacingd(
#                     keys=["image", "label"],
#                     pixdim=(1.5, 1.5, 2.0),
#                     mode=("bilinear", "nearest"),
#                 ),
#                 transforms.CropForegroundd(
#                     keys=["image", "label"],
#                     source_key="label",  # 使用label作为前景参考
#                     select_fn=lambda x: x > 0,  # 选择标签值>0的区域
#                     margin=(5, 5, 5),  # 可选：在肿瘤周围添加边缘
#                     ),
#                 transforms.Resized(keys=["image", "label"], spatial_size=config['image_size']),
#             ]
    # enddd
[
#        # 原有的预处理流程保持不变
#         transforms.LoadImaged(keys=["image", "label"]),
#         transforms.EnsureChannelFirstd(keys=["image", "label"]),
#         transforms.ScaleIntensityRanged(
#             keys=["image"],
#             a_min=-175,
#             a_max=250,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),
#         transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#         transforms.Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.5, 1.5, 2.0),
#             mode=("bilinear", "nearest"),
#         ),
#         transforms.CropForegroundd(
#             keys=["image", "label"],
#             source_key="label",
#             select_fn=lambda x: x > 0,
#             margin=(5, 5, 5),
#         ),
#         # 添加中心裁剪确保输入尺寸一致（可选）
#         transforms.CenterSpatialCropd(
#             keys=["image", "label"],
#             roi_size=[96, 96, 96]  # 根据实际数据调整
#         ),
#         transforms.Resized(
#             keys=["image", "label"],
#             spatial_size=config['image_size'],  # 确保是(64,64,64)
#             mode=("bilinear", "nearest")
#         ),
# 1. 加载数据（保持原始分辨率）
    transforms.LoadImaged(keys=["image", "label"]),
    # 2. 添加通道维度
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    # 3. 调整图像强度范围（不影响空间结构）
    # transforms.ScaleIntensityRanged(
    #     keys=["image"],
    #     a_min=-175,
    #     a_max=250,
    #     b_min=0.0,
    #     b_max=1.0,
    #     clip=True,
    # ),
    # 4. 统一坐标系（仅调整轴顺序，不重采样）
    # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    # 5. 根据标签前景裁剪，确保包含癌症区域
    # transforms.CropForegroundd(
    #     keys=["image", "label"],
    #     source_key="label",
    #     select_fn=lambda x: x > 0,
    #     margin=32,  # 扩展边界，确保裁剪后至少 64×64×64
    # ),

    # 强制填充到至少 64×64×64（若裁剪后尺寸不足）
    # transforms.SpatialPadd(
    #     keys=["image", "label"],
    #     spatial_size=(64, 64, 64),
    #     mode="constant",  # 填充常数值（如 0）
    # ),

    # 中心裁剪到固定尺寸 64×64×64
    # transforms.CenterSpatialCropd(
    #     keys=["image", "label"],
    #     roi_size=(64, 64, 64)
    # ),
]

    )

# +
batch_size = config['batch_size']

train_ds = DecathlonDataset(
    root_dir=root_dir,
    task=config['dataset'],
    section="training",
    cache_rate=1.0,
    num_workers=4,
    download=False,
    seed=0,
    transform=train_transforms,
)
print(f"Length of training data: {len(train_ds)}")
print(f'Train image shape {train_ds[0]["image"].shape}')
print(f'Train label shape {train_ds[0]["label"].shape}')
logger.info(f"Length of training data: {len(train_ds)}")
logger.info(f"Train image shape: {train_ds[0]['image'].shape}")
logger.info(f"Train label shape: {train_ds[0]['label'].shape}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True
)
# -

# ## Prepare validation dataset

# +
val_ds = DecathlonDataset(
    root_dir=root_dir,
    task=config['dataset'],
    section="validation",
    cache_rate=1.0,
    num_workers=4,
    download=False,
    seed=0,
    transform=train_transforms,
)
print(f"Length of validation data: {len(val_ds)}")
print(f'Validation Image shape {val_ds[0]["image"].shape}')
print(f'Validation Label shape {val_ds[0]["label"].shape}')
logger.info(f"Length of validation data: {len(val_ds)}")
logger.info(f"Validation Image shape: {val_ds[0]['image'].shape}")
logger.info(f"Validation Label shape: {val_ds[0]['label'].shape}")

val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True
)

# Save sample data visualizations for inspection
import matplotlib.pyplot as plt
import os




# Set saving path
save_dir = result_image_dir + "train_samples"
os.makedirs(save_dir, exist_ok=True)


import numpy as np
import nibabel as nib
import torch

def save_as_nii_gz(sample, save_name="."):
    image = sample["image"]
    label = sample["label"]

    # 处理图像数据
    image_np = image.squeeze().cpu().detach().numpy()  # 移除通道/批次维度
    # 调整轴顺序（假设原始维度为 [D, H, W]，转为 [W, H, D]）
    image_np = np.transpose(image_np, (2, 1, 0))  # 根据实际情况调整转置
    affine = np.eye(4)  # 仿射矩阵（假设为单位矩阵）
    img_nifti = nib.Nifti1Image(image_np.astype(np.float32), affine)
    nib.save(img_nifti, os.path.join(save_dir,f"{save_name}image.nii.gz"))

    # 处理标签数据
    label_np = label.squeeze().cpu().detach().numpy()
    label_np = np.transpose(label_np, (2, 1, 0))  # 同样调整轴顺序
    label_nifti = nib.Nifti1Image(label_np.astype(np.int16), affine)
    nib.save(label_nifti, os.path.join(save_dir,f"{save_name}label.nii.gz"))



# Get samples for visualization
num_samples = 5

for i in range(num_samples):
    sample = train_ds[i]
    # 存下niigz
    # save_as_nii_gz(sample, save_name=f"sample{i}")
    
    image = sample["image"]
    label = sample["label"]

    
    # Select central slices from each dimension for visualization
    # Axial view (top-down)
    slice_idx_z = image.shape[3] // 2
    # Coronal view (front-back)
    slice_idx_y = image.shape[2] // 2
    # Sagittal view (left-right)
    slice_idx_x = image.shape[1] // 2
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    # Axial slices
    axes[0, 0].imshow(image[0, :, :, slice_idx_z], cmap="gray")
    axes[0, 0].set_title(f"Image {i} (Axial)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(label[0, :, :, slice_idx_z], cmap="jet", alpha=0.7)
    axes[0, 1].set_title(f"Label {i} (Axial)")
    axes[0, 1].axis("off")
    
    # Coronal slices
    axes[1, 0].imshow(image[0, :, slice_idx_y, :], cmap="gray")
    axes[1, 0].set_title(f"Image {i} (Coronal)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(label[0, :, slice_idx_y, :], cmap="jet", alpha=0.7)
    axes[1, 1].set_title(f"Label {i} (Coronal)")
    axes[1, 1].axis("off")
    
    # Sagittal slices
    axes[2, 0].imshow(image[0, slice_idx_x, :, :], cmap="gray")
    axes[2, 0].set_title(f"Image {i} (Sagittal)")
    axes[2, 0].axis("off")
    
    axes[2, 1].imshow(label[0, slice_idx_x, :, :], cmap="jet", alpha=0.7)
    axes[2, 1].set_title(f"Label {i} (Sagittal)")
    axes[2, 1].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_{i}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Saved: {save_path}")
# -

#
# ## Define 3D network, scheduler, optimizer, and inferer
#
# We update the UNet to use 3D spatial dimensions

device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,  # Changed to 3D
    in_channels=2,   # 1 image channel + 1 noisy label channel
    out_channels=1,  # Output is the predicted noise
    num_channels=config['unet_channels'],
    attention_levels=config['attention_levels'],
    num_res_blocks=config['num_res_blocks'],
    num_head_channels=64,
    with_conditioning=False,
)


model_path = "/home/yw4445/GenerativeModelTutorialOutput/Task06_Lung/3D_runs_2025-04-14_15-09-47/segmodel_epoch4050.pt"
# model.load_state_dict(torch.load(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scaler.load_state_dict(checkpoint['scaler_state_dict'])
# start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
logger.info(f'pretrain from {model_path}')

model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])  # Slightly lower learning rate for 3D
optimizer = torch.optim.AdamW(model.parameters(), 1e-4, amsgrad = True)
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-6, weight_decay=1e-4)
inferer = DiffusionInferer(scheduler)

#
# ### Model training of the 3D Diffusion Model

n_epochs = config['epochs']
val_interval = 50
epoch_loss_list = []
val_epoch_loss_list = []

# +
scaler = GradScaler()
total_start = time.time()

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0

    for step, data in enumerate(train_loader):
        images = data["image"].to(device)
        seg = data["label"].to(device)  # 3D ground truth segmentation
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(seg).to(device)
            noisy_seg = scheduler.add_noise(
                original_samples=seg, noise=noise, timesteps=timesteps
            )  # Add noise to the segmentation mask
            
            # Concatenate image and noisy segmentation
            combined = torch.cat((images, noisy_seg), dim=1)
            
            # Model prediction
            prediction = model(x=combined, timesteps=timesteps)
            
            # Calculate loss
            loss = F.mse_loss(prediction.float(), noise.float())
            
        scaler.scale(loss).backward()
        # 添加梯度裁剪（最大范数设为1.0）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    epoch_loss_list.append(epoch_loss / (step + 1))
    logger.info("Epoch %d Training loss %f", epoch, epoch_loss / (step + 1))
    
    if (epoch) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        
        for step, data_val in enumerate(val_loader):
            images = data_val["image"].to(device)
            seg = data_val["label"].to(device)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)
            
            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(seg).to(device)
                    noisy_seg = scheduler.add_noise(original_samples=seg, noise=noise, timesteps=timesteps)
                    combined = torch.cat((images, noisy_seg), dim=1)
                    prediction = model(x=combined, timesteps=timesteps)
                    
                    val_loss = F.mse_loss(prediction.float(), noise.float())
                    
            val_epoch_loss += val_loss.item()
            
        print("Epoch", epoch, "Validation loss", val_epoch_loss / (step + 1))
        logger.info("Epoch %d Validation loss %f", epoch, val_epoch_loss / (step + 1))
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))
        
        # Save model checkpoint
        if (epoch) % (val_interval) == 0:
            # 保存检查点（需在原有代码中添加）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # 混合精度训练需保存GradScaler状态
                'loss': epoch_loss,
            }
            torch.save(checkpoint, result_image_dir + f"segmodel_epoch{epoch}.pt")
                        
            
            # torch.save(model.state_dict(), result_image_dir + f"segmodel_epoch{epoch}.pt")

# Save final model
torch.save(model.state_dict(), result_image_dir + "segmodel_final.pt")
total_time = time.time() - total_start
print(f"Train diffusion completed, total time: {total_time}.")
logger.info("Train diffusion completed, total time: %s.", total_time)

# Plot learning curves
plt.style.use("seaborn-v0_8-bright")
plt.figure(figsize=(10, 6))
plt.title("Learning Curves 3D Diffusion Model", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig(result_image_dir + "learning_curve.png", dpi=300, bbox_inches="tight") 
plt.show()
# -

#
# # 3D Sampling from the trained diffusion model
# Here we generate 3D segmentation masks for input volumes


import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_all_slices(input_volume, input_label, save_path=None):
    """
    Visualize all slices of a 3D volume and corresponding label maps side by side.
    
    Parameters:
    - input_volume: 3D volume data with shape (H, W, D) or (C, H, W, D)
    - input_label: 3D label data with same shape as input_volume
    - save_path: Optional path to save the figure
    """
    # Handle different input dimensions
    if len(input_volume.shape) == 4:  # (C, H, W, D)
        volume = input_volume[0]  # Take first channel
    else:
        volume = input_volume
    
    # Get number of slices
    num_slices = volume.shape[2]
    
    # Calculate grid dimensions (trying to make it roughly square)
    grid_size = math.ceil(math.sqrt(num_slices))
    rows = grid_size
    cols = math.ceil(num_slices / rows)
    
    # Create figure with enough space
    plt.figure(figsize=(cols*4, rows*4))
    
    # Plot all slices
    for i in range(num_slices):
        # Plot input volume
        plt.subplot(rows, cols*2, i*2+1)
        plt.imshow(volume[:, :, i], cmap="gray")
        plt.title(f"Image Slice {i+1}")
        plt.axis("off")
        
        if input_label != None:
            if len(input_label.shape) == 4:  # (C, H, W, D)
                label = input_label[0]  # Take first channel
            else:
                label = input_label
    
            # Plot label
            plt.subplot(rows, cols*2, i*2+2)
            plt.imshow(label[:, :, i], cmap="gray")
            plt.title(f"Label Slice {i+1}")
            plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(result_image_dir + save_path, dpi=150, bbox_inches="tight")
    
    plt.show()




# +
# Select a test volume
idx = 0
data = val_ds[idx]
input_volume = data["image"]
input_label = data["label"]

# # Visualize central slices of the input volume and ground truth
# slice_idx_z = input_volume.shape[3] // 2

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(input_volume[0, :, :, slice_idx_z], cmap="gray")
# plt.title("Input Image (Axial)")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(input_label[0, :, :, slice_idx_z], cmap="gray")
# plt.title("Ground Truth (Axial)")
# plt.axis("off")

# plt.tight_layout()
# plt.savefig(result_image_dir + "sample_input_3d.png", dpi=300, bbox_inches="tight") 
# plt.show()
visualize_all_slices(input_volume, input_label, save_path="sample_input_3d_all_slices_visualization.png")

model.eval()

# Generate an ensemble of segmentations
n = 5  # Number of samples in ensemble
inputimg = data["image"][0, ...]
input_vol = inputimg[None, None, ...].to(device)
ensemble = []

for k in range(n):
    logger.info(f"Generating sample {k+1}/{n}")
    
    # Initialize with random noise
    noise = torch.randn_like(input_vol).to(device)
    current_vol = noise  # Start with random noise for segmentation mask
    
    # Concatenate the input volume with noise
    combined = torch.cat((input_vol, noise), dim=1)

    # Setup sampling with 1000 steps
    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps)
    chain = torch.zeros(current_vol.shape)
    # Generate segmentation through iterative denoising
    for t in progress_bar:
        with autocast(enabled=False):
            with torch.no_grad():
                # Get model prediction
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                
                # Update current volume using scheduler
                current_vol, _ = scheduler.step(model_output, t, current_vol)
                if t % 100 == 0:
                    chain = torch.cat((chain, current_vol.cpu()), dim=-1)
                # Update combined input for next step
                combined = torch.cat((input_vol, current_vol), dim=1)
    
    visualize_all_slices(current_vol[0, 0, :, :,:].cpu(), None, save_path=f"denoised_mask_3d_{k}.png")
    # # Visualize the result (central axial slice)
    # plt.style.use("default")
    # plt.imshow(chain[0, 0, :, :, slice_idx_z].cpu(), vmin=0, vmax=1, cmap="gray")
    # plt.tight_layout()
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(result_image_dir + f"denoised_mask_3d_{k}.png", dpi=300, bbox_inches="tight") 
    # plt.show()
    
    # Add to ensemble
    ensemble.append(current_vol)

# Helper function for 3D Dice score
def dice_score_3d(pred, target, empty_score=1.0):
    pred = pred.detach().cpu().numpy().astype(bool)
    target = target.detach().cpu().numpy().astype(bool)
    
    im_sum = pred.sum() + target.sum()
    if im_sum == 0:
        return empty_score
    
    # Compute Dice coefficient
    intersection = np.logical_and(pred, target).sum()
    return 2.0 * intersection / im_sum

# Calculate individual dice scores
for i in range(len(ensemble)):
    prediction = torch.where(ensemble[i] > 0.5, 1, 0).float()
    score = dice_score_3d(prediction[0, 0], input_label[0])
    print(f"Dice score of sample {i}: {score}")
    logger.info(f"Dice score of sample {i}: {score}")

# Calculate ensemble statistics
E = torch.stack([torch.where(vol > 0.5, 1, 0).float() for vol in ensemble])
var = torch.var(E, dim=0)  # Variance map
mean = torch.mean(E, dim=0)  # Mean map
mean_prediction = torch.where(mean > 0.5, 1, 0).float()

# Dice score on the ensemble mean
score = dice_score_3d(mean_prediction[0, 0], input_label[0])
print(f"Dice score on the ensemble mean: {score}")
logger.info(f"Dice score on the ensemble mean: {score}")

# Visualize mean and variance maps (central slice)
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_volume[0, :, :, slice_idx_z].cpu(), cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mean[0, 0, :, :, slice_idx_z].cpu(), cmap="gray")
plt.title("Mean Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(var[0, 0, :, :, slice_idx_z].cpu(), cmap="jet")
plt.title("Prediction Variance")
plt.axis("off")

plt.tight_layout()
plt.savefig(result_image_dir + "ensemble_results_3d.png", dpi=300, bbox_inches="tight")
plt.show()

# Save 3D visualization across multiple slices
num_vis_slices = 5
slice_indices = np.linspace(0, input_volume.shape[3]-1, num_vis_slices).astype(int)

plt.figure(figsize=(15, 10))
for i, s_idx in enumerate(slice_indices):
    # Input volume
    plt.subplot(3, num_vis_slices, i + 1)
    plt.imshow(input_volume[0, :, :, s_idx].cpu(), cmap="gray")
    if i == 0:
        plt.ylabel("Input", fontsize=14)
    plt.title(f"Slice {s_idx}")
    plt.axis("off")
    
    # Ground truth
    plt.subplot(3, num_vis_slices, i + num_vis_slices + 1)
    plt.imshow(input_label[0, :, :, s_idx].cpu(), cmap="gray")
    if i == 0:
        plt.ylabel("Ground Truth", fontsize=14)
    plt.axis("off")
    
    # Prediction
    plt.subplot(3, num_vis_slices, i + 2*num_vis_slices + 1)
    plt.imshow(mean_prediction[0, 0, :, :, s_idx].cpu(), cmap="gray")
    if i == 0:
        plt.ylabel("Prediction", fontsize=14)
    plt.axis("off")

plt.tight_layout()
plt.savefig(result_image_dir + "3d_slices_comparison.png", dpi=300, bbox_inches="tight")
plt.show()