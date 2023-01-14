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