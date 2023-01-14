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
### [Compute_data_norm.py](https://github.com/come880412/DL_common/blob/main/Compute_data_norm.py)
Getting normalized mean and std for an image dataset
```bash
python Compute_data_norm.py <image_dir> <image_size>
```
- image_dir: Path to image dataset
- image_size: Resize image to the given size parameter

### [LinearWarmupCosineAnnealingLR.py](https://github.com/come880412/DL_common/blob/main/LinearWarmupCosineAnnealingLR.py)
"LinearWarmupCosineAnnealing" Learning rate scheduler trick decaying by epoch
```bash
python LinearWarmupCosineAnnealingLR.py <warmup_epochs> <max_epochs> <warmup_start_lr> <eta_min>
```
- warmup_epochs: Maximum number of epochs for linear warmup
- max_epochs: Maximum number of epochs
- warmup_start_lr: Learning rate to start the linear warmup. Default: 0.
- eta_min: Minimum learning rate. Default: 0.

<p align="center">
<img src="https://github.com/come880412/DL_common/blob/main/images/lr_decay.png" width=40% height=40%>
</p>