## Lung Ablation Zone Prediction (3D UNet baseline + 3D DDPM)

### Quickstart

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Prepare a CSV listing your samples (train/val/test):
```csv
ct_path,tumor_mask_path,target_mask_path,probe_mask_path,energy,time
/data/ct_001.nii.gz,/data/tumor_001.nii.gz,/data/abl_001.nii.gz,,60,600
```
Notes:
- `probe_mask_path` can be empty if not available.
- `energy` and `time` are optional scalars; leave empty to skip.

3) Edit `configs/train_unet.yaml` or `configs/train_ddpm.yaml` to point to your CSVs and set training hyperparameters.

4) Train the baseline UNet:
```bash
python -m ablation_pred.training.train_unet --config configs/train_unet.yaml
```

5) Train the DDPM:
```bash
python -m ablation_pred.training.train_ddpm --config configs/train_ddpm.yaml
```

6) Evaluate:
```bash
python -m ablation_pred.eval.evaluate --config configs/train_unet.yaml --ckpt runs/unet/last.ckpt
```

### Inputs and Outputs
- Inputs: pre-ablation CT (1 channel), tumor mask (1), optional probe mask (1), optional scalar conditioning (energy/time) broadcast to volumes.
- Outputs:
  - UNet: deterministic ablation probability map (sigmoid) -> threshold to get mask.
  - DDPM: probabilistic ablation map via sampling; average multiple samples for mean prediction; quantify uncertainty with per-voxel variance.

### Notes
- Voxel spacing and patch size are configured in YAML; data are resampled on the fly.
- Metrics include Dice, HD95, volume error, tumor coverage and 5–10 mm margin coverage.
# Diffusion Models for Implicit Image Segmentation Ensembles

We provide the official Pytorch implementation of the paper [Diffusion Models for Implicit Image Segmentation Ensembles](https://arxiv.org/abs/2112.03145) by Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe Valmaggia, and Philippe C. Cattin.

The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

## Paper Abstract

Diffusion models have shown impressive performance for generative modelling of images. In this paper, we present a novel semantic segmentation method based on diffusion models. By modifying the training and sampling scheme, we show that diffusion models can perform lesion segmentation of medical images. To generate an image specific segmentation, we train the model on the ground truth segmentation, and use the image as a prior during training and in every step during the sampling process. With the given stochastic sampling process, we can generate a distribution of segmentation masks. This property allows us to compute pixel-wise uncertainty maps of the segmentation, and allows an implicit ensemble of segmentations that increases the segmentation performance. We evaluate our method on the BRATS2020 dataset for brain tumor segmentation. Compared to state-of-the-art segmentation models, our approach yields good segmentation results and, additionally, detailed uncertainty maps.


## Data

We evaluated our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html).
For our dataloader, which can be found in the file *guided_diffusion/bratsloader.py*, the 2D slices need to be stored in the following structure:

```
data
└───training
│   └───slice0001
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│       │   seg.nii.gz
│   └───slice0002
│       │  ...
└───testing
│   └───slice1000
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│   └───slice1001
│       │  ...

```

A mini-example can be found in the folder *data*.
If you want to apply our code to another dataset, make sure the loaded image has attached the ground truth segmentation as the last channel.


## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```
To train the segmentation model, run

```
python3 scripts/segmentation_train.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
For sampling an ensemble of 5 segmentation masks with the DDPM approach, run:

```
python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated segmentation masks will be stored in the *results* folder. A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).

## Citation
If you use this code, please cite

```
@misc{wolleb2021diffusion,
      title={Diffusion Models for Implicit Image Segmentation Ensembles}, 
      author={Julia Wolleb and Robin Sandkühler and Florentin Bieder and Philippe Valmaggia and Philippe C. Cattin},
      year={2021},
      eprint={2112.03145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
