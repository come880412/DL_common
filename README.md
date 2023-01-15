# DL_common
Implement some useful tools for deep learning

## Enviroments
I use `python3.9` and `torch==1.3.0` in Linux system.
```bash
# You can create your own conda enviroment as follows
conda create -n <name> python=3.9
conda activate <name>
pip install -r requirements.txt
```

## Implementations
### [Compute_data_norm.py](https://github.com/come880412/DL_common/blob/main/scripts/Compute_data_norm.py)
Getting normalized mean and std for an image dataset.
```bash
python Compute_data_norm.py <image_dir> <image_size>
```
- image_dir: Path to image dataset
- image_size: Resize image to the given size parameter

### [LinearWarmupCosineAnnealingLR.py](https://github.com/come880412/DL_common/blob/main/scripts/LinearWarmupCosineAnnealingLR.py)
"LinearWarmupCosineAnnealing" Learning rate scheduler trick decaying by epoch.
```bash
python LinearWarmupCosineAnnealingLR.py <warmup_epochs> <max_epochs> <warmup_start_lr> <eta_min>
```
- warmup_epochs: Maximum number of epochs for linear warmup. Default:10
- max_epochs: Maximum number of epochs. Default:100
- warmup_start_lr: Learning rate to start the linear warmup. Default: 0.
- eta_min: Minimum learning rate. Default: 0.

<p align="center">
<img src="https://github.com/come880412/DL_common/blob/main/images/lr_decay.png" width=60% height=60%>
</p>

### [RandomSeed.py](https://github.com/come880412/DL_common/blob/main/scripts/RandomSeed.py)
Fix random seed for program reproducible.
```bash
python RandomSeed.py <seed>
```
- seed: Set random seed what you want

### [Metric.py](https://github.com/come880412/DL_common/blob/main/scripts/Metric.py)
Compute acc, auc, precision, recall, and F1-score.
```bash
python Metric.py
```

### [Make_noisy_image.py](https://github.com/come880412/DL_common/blob/main/scripts/Metric.py)
Reference: https://github.com/hendrycks/robustness \
Make noise on an image, supporting the following noisy types: \
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog,
brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate
```bash
python Make_noisy_image.py <data_dir> <save_dir> <severity>
```
- data_dir: Path to image dataset
- save_dir: Save image path
- severity: noisy level

<p align="center">
<img src="https://github.com/come880412/DL_common/blob/main/images/severity_5.png" width=60% height=60%>
</p>