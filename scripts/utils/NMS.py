"""
Implement NMS algorithm for object detection
python scripts/utils/NMS.py
"""

import glob
import argparse
import os
import numpy as np
from scripts.utils.Compute_IoU import bboxes_iou

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4], format="xyrb")
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_path', required=True, help='Path to the folder containing your detected bounding boxes')
    args = parser.parse_args()

    preds = np.concatenate([np.loadtxt(f, dtype='str') for f in glob.glob(os.path.join(args.det_path, "*.txt"))])
    preds_new = []
    for pred in preds:
        preds_new.append([int(pred[2]), int(pred[3]), int(pred[2]) + int(pred[4]), int(pred[3]) + int(pred[5]), float(pred[1]), 0])

    preds_new = np.array(preds_new)
    print("# of bboxes before NMS: ", len(preds))
    preds_new = nms(preds_new, iou_threshold=0.5, sigma=0.3, method="nms")
    print("# of bboxes after NMS: ", len(preds_new))