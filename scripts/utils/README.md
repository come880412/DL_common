## Utils
This section implments some useful deep learning tools.

## Implmentations
### [RandomSeed.py](https://github.com/come880412/DL_common/blob/main/scripts/utils/RandomSeed.py)
Fix random seed for program reproducible.
```bash
python scripts/utils/RandomSeed.py <seed>
```
- seed: Set random seed what you want

### [Metric.py](https://github.com/come880412/DL_common/blob/main/scripts/utils/Metric.py)
Compute acc, auc, precision, recall, and F1-score.
```bash
python scripts/utils/Metric.py
```


### [Compute_IoU.py](https://github.com/come880412/DL_common/blob/main/scripts/utils/Compute_IoU.py)
Compute the IOU score of the given two bboxes
```bash
python scripts/utils/Compute_IoU.py <det_path> <gt_path> <format>
```
- det_path: Path to the folder containing your detected bounding boxes
- gt_path: Path to the folder containing your ground truth bounding boxes
- format: Format of the coordinates of the bounding boxes ('xywh': <left> <top> <width> <height> | 'xyrb': <left> <top> <right> <bottom>)

### [NMS.py](https://github.com/come880412/DL_common/blob/main/scripts/utils/NMS.py)
Implement NMS algorithm for object detection
```bash
python scripts/utils/NMS.py <det_path>
```
- det_path: Path to the folder containing your detected bounding boxes