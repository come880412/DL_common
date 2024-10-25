import numpy as np
import os
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import Record, Metric, fixed_seed
from config.config import load_config
from dataset.Dataset import get_dataloader
from models.build_model import build_model

def test(args, model ,loader, criterion, device):
    model.eval()
    record = Record(args)
    metric = Metric(args)
    label_to_category = record.label_to_category

    classes = []
    for category in label_to_category.values():
        classes.append(category)

    pbar = tqdm.tqdm(total=len(loader), ncols=0, desc=f"{args.test_mode}", unit="step")
    loss_total = 0.
    num_data = 0
    csv_save = [['image_name', 'pred', "label"]]
    pred_score, pred_label = [], []
    with torch.no_grad():
        for image, label, image_name in loader:
            image, label = image.to(device), label.to(device)

            pred = model(image)
            loss = criterion(pred, label)
            loss_total += loss * len(image)
            num_data += len(image)

            pbar.update()
            pbar.set_postfix(
                loss=f"{(loss_total / num_data):.4f}",
            )
            record.update(pred, label, image_name)
            for score in pred.cpu().detach():
                score = F.softmax(score).numpy()
                pred_score.append(score[1])

    metric_dict, acc = metric.compute_metric(record.y_pred, record.y_true)
    pbar.close()

    # Plot confusion matrix
    cm = confusion_matrix(record.y_true, record.y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    print("\n--------- Save confussion matrix ----------")
    plt.savefig(os.path.join(args.output_path, f"cm_{args.test_mode}.png"))

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(record.y_true, np.array(pred_score))
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Cat-dog classification')
    display.plot()
    print("\n--------- Save ROC curve ----------")
    plt.savefig(os.path.join(args.output_path, f"ROC_curve_{args.test_mode}.png"))
    
    for image_name, pred, label in zip(record.image_name, record.y_pred, record.y_true):
        csv_save.append([image_name, pred, label])
    np.savetxt(os.path.join(args.output_path, f"output_{args.test_mode}.csv"), csv_save, encoding='utf-8-sig', fmt='%s', delimiter=',')

    print("\n---- Prediction ----")
    print(metric_dict)
    print(f"Accuracy: {acc:.3f}%")

if __name__ == '__main__':
    args = load_config()
    fixed_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader, test_loader, n_classes = get_dataloader(args)
    loader = val_loader if args.test_mode == "val" else test_loader

    model = build_model(args, device, n_classes)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f'Loading pretrained model from {args.resume}')
    checkpoints = torch.load(args.resume)
    model.load_state_dict(checkpoints["model_state_dict"])

    test(args, model, loader, criterion, device)