"""
Compute the IOU score of the given two bboxes
python scripts/utils/Compute_IoU.py
"""

import glob
import argparse
import os
import numpy as np

def bboxes_iou(boxes1, boxes2, format="xywh"): 
    """
    boxes1: The ground-truth box with the content [x_min, y_min, width, height]
    boxes2: The predicted box with the content [x_min, y_min, width, height]
    format: Format of the coordinates of the bounding boxes ('xywh': <left> <top> <width> <height> | 'xyrb': <left> <top> <right> <bottom>)
    """

    boxes1 = np.array(boxes1, dtype=int)
    boxes2 = np.array(boxes2, dtype=int)

    if format == "xywh":
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right = np.minimum(boxes1[..., 0] + boxes1[..., 2], boxes2[..., 0] + boxes2[..., 2])
        down = np.minimum(boxes1[..., 1] + boxes1[..., 3], boxes2[..., 1] + boxes2[..., 3])
        right_down    = np.array([right, down])
    elif format == "xyrb":
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    else:
        raise NameError("Please give the bbox format('xywh' or 'xyrb')")

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_path', required=True, help='Path to the folder containing your detected bounding boxes')
    parser.add_argument('--gt_path', required=True, help='Path to the folder containing your ground truth bounding boxes')
    parser.add_argument('--format', default="xywh", help='Format of the coordinates of the bounding boxes'
                                                        '(\'xywh\': <left> <top> <width> <height> | \'xyrb\': <left> <top> <right> <bottom>)')
    args = parser.parse_args()

    pred_lists = glob.glob(os.path.join(args.det_path, "*.txt"))
    gth_lists = glob.glob(os.path.join(args.gt_path, "*.txt"))
    pred_lists.sort()
    gth_lists.sort()

    pred_boxes = np.loadtxt(pred_lists[0], dtype='str')
    gth_boxes = np.loadtxt(gth_lists[0], dtype='str')
    
    iou_list = []
    for pred_box in pred_boxes:
        iou = [bboxes_iou(gth_box[-4:], pred_box[-4:]) for gth_box in gth_boxes]
        iou_list.append(iou)
    print("Predicted bboxes: ", pred_boxes)
    print("Ground truth bboxes: ", gth_boxes)
    print("IoU score: ", iou_list)
