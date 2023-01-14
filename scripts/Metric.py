"""
Compute acc, auc, precision, recall, and F1-score.
python Metric.py
"""


import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

class Metric():
    def __init__(self):
        pass
    def get_acc(self, 
                y_pred: np.array, 
                y_true: np.array, 
                task: str, 
                threshold: float =0.5
                ):
        """ ACC metric
        y_pred: the predicted result of each class, shape: (num_data, num_classes)
        y_true: the ground truth labels, shape: (num_data,) for 'multi-class' or (num_data, n_classes) for 'multi-label'
        task: Task for current dataset ['multi-class', 'multi-label']
        threshold: the threshold for multi-label task
        """
        
        if task == 'multi-class':
            y_pred = np.argmax(y_pred, axis=1)
            correct = np.sum(np.equal(y_true, y_pred))
            total = y_true.shape[0]
        elif task == 'multi-label':
            correct_each_class = []
            total_each_class = []
            y_pred = y_pred > threshold

            for label in range(y_true.shape[1]):
                correct_each_class.append(np.sum(np.equal(y_true[:, label], y_pred[:, label])))
                total_each_class.append(y_true.shape[0])
            correct = sum(correct_each_class)
            total = sum(total_each_class)
        
        return correct / total
    
    def get_auc(self, 
                y_pred: np.array, 
                y_true: np.array, 
                task: str
                ):
        '''AUC metric.
        y_pred: the predicted result of each class, shape: (num_data, n_classes)
        y_true: the ground truth labels, shape: (num_data,) for 'multi-class' or (num_data, n_classes) for 'multi-label'
        task: Task for current dataset ['multi-class', 'multi-label']
        '''
        
        if task == 'multi-class':
            auc = 0
            for i in range(y_pred.shape[1]):
                y_true_binary = (y_true == i).astype(float)
                y_score_binary = y_pred[:, i]
                auc += roc_auc_score(y_true_binary, y_score_binary)
            auc = auc / y_pred.shape[1]
        elif task == 'multi-label':
            auc = 0
            for i in range(y_pred.shape[1]):
                label_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc += label_auc
            auc = auc / y_pred.shape[1]
        return auc

    def get_PRF(self, 
                y_pred: np.array, 
                y_true: np.array, 
                task: str, 
                threshold: float =0.5
                ):
        """ Precision_Recall_F1score metrics
        y_pred: the predicted result of each class, shape: (num_data, num_classes)
        y_true: the ground truth labels, shape: (num_data,) for 'multi-class' or (num_data, n_classes) for 'multi-label'
        task: Task for current dataset ['multi-class', 'multi-label']
        threshold: the threshold for multi-label task
        """
        eps=1e-20
        if task == 'multi-label':
            # y_true = y_true.type(torch.cuda.ByteTensor)
            # y_pred = y_pred.type(torch.cuda.FloatTensor).sigmoid()

            precision_total = 0
            recall_total = 0
            f1_total = 0
            for i in range(y_pred.shape[1]):
                output_class = y_pred[:, i]
                target_class = y_true[:, i]
                
                prob = output_class > threshold
                label = target_class > 0.5
                # print(prob, label)
                TP = np.float16((prob & label).sum())
                TN = np.float16(((~prob) & (~label)).sum())
                FP = np.float16((prob & (~label)).sum())
                FN = np.float16(((~prob) & label).sum())

                precision = TP / (TP + FP + eps)
                recall = TP / (TP + FN + eps)

                result_f1 = 2 * precision  * recall / (precision + recall + eps)

                precision_total += precision.item()
                recall_total += recall.item()
                f1_total += result_f1.item()

            f1_avg = f1_total / y_pred.shape[1]
            precision_avg = precision_total / y_pred.shape[1]
            recall_avg = recall_total / y_pred.shape[1]

            return f1_avg, precision_avg, recall_avg
        elif task == 'multi-class':
            y_pred = np.argmax(y_pred, axis=1)

            # y_pred = y_pred.cpu().detach().numpy()
            # y_true = y_true.cpu().detach().numpy()

            confusion = confusion_matrix(y_true, y_pred)
            precision_total = 0
            recall_total = 0
            f1_total = 0
            for i in range(len(confusion)):
                TP = confusion[i, i]
                FP = sum(confusion[:, i]) - TP
                FN = sum(confusion[i, :]) - TP

                precision = TP / (TP + FP + eps)
                recall = TP / (TP + FN + eps)
                result_f1 = 2 * precision  * recall / (precision + recall + eps)

                precision_total += precision
                recall_total += recall
                f1_total += result_f1

            f1_avg = f1_total / len(confusion)
            precision_avg = precision_total / len(confusion)
            recall_avg = recall_total / len(confusion)

            return f1_avg, precision_avg, recall_avg


def main():
    metric = Metric()

    print("------ Multi-label classification probloem ------")
    y_true = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.float16) 
    y_pred = torch.randn(4, 4)
    print("Predicted: ", y_pred.sigmoid() > 0.5)
    print("Ground-truth: ", y_true)

    acc = metric.get_acc(y_pred.sigmoid().cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-label')
    auc = metric.get_auc(y_pred.sigmoid().cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-label')
    f1, Precision, Recall = metric.get_PRF(y_pred.sigmoid().cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-label', 0.5)
    print(f"Macro Acc: {acc*100:.2f}%")
    print(f"Area Under Curve (AUC): {auc*100:.2f}%")
    print(f"Precision: {Precision:.3f}")
    print(f"Recall: {Recall:.3f}")
    print(f"f1-score: {f1:.3f}")

    print("\n------ Multi-class classification probloem ------")
    """multi-class acc test"""
    y_true = torch.tensor([0, 1, 1, 0, 2, 1, 3, 4, 0])
    y_pred = torch.randn(len(y_true), 5)
    print("Predicted: ", torch.argmax(y_pred, dim=1))
    print("Ground-truth: ", y_true)

    acc = metric.get_acc(y_pred.softmax(dim=-1).cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-class')
    auc = metric.get_auc(y_pred.softmax(dim=-1).cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-class')
    f1, Precision, Recall = metric.get_PRF(y_pred.softmax(dim=-1).cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'multi-class')
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Area Under Curve (AUC): {auc*100:.2f}%")
    print(f"Precision: {Precision:.3f}")
    print(f"Recall: {Recall:.3f}")
    print(f"f1-score: {f1:.3f}")
    # print(precision_recall_fscore_support(y_true.numpy(), np.argmax(y_pred.numpy(), axis=1), average='macro'))


if __name__ == "__main__":
    main()