## Data
This section implements some useful tools for processing data.

## Implementation
### [Compute_data_norm.py](https://github.com/come880412/DL_common/blob/main/scripts/data/Compute_data_norm.py)
Getting normalized mean and std for an image dataset.
```bash
python scripts/data/Compute_data_norm.py <image_dir> <image_size>
```
- image_dir: Path to image dataset
- image_size: Resize image to the given size parameter


### [Make_noisy_image.py](https://github.com/come880412/DL_common/blob/main/scripts/data/Make_noisy_image.py)
Reference: https://github.com/hendrycks/robustness \
Make noise on an image, supporting the following noisy types: \
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog,
brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate
```bash
python scripts/data/Make_noisy_image.py <data_dir> <save_dir> <severity>
```
- data_dir: Path to image dataset
- save_dir: Save image path
- severity: noisy level

<p align="center">
<img src="https://github.com/come880412/DL_common/blob/main/images/severity_5.png" width=60% height=60%>
</p>