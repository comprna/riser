import sys
from statistics import mean
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchinfo import summary

from data import SignalDataset
from resnet import ResNet
from utilities import get_config


def count_correct(y_true, y_pred):
    assert len(y_pred) == len(y_true)
    n_correct = 0
    for i, pred in enumerate(y_pred):
        if pred == y_true[i]:
            n_correct += 1
    return n_correct


def main():

    # CL args

    # model_file = sys.argv[1]
    # data_dir = sys.argv[2]
    # batch_size = int(sys.argv[3])
    model_file = './local_data/models/train-resnet-33_0_best_model.pth'
    data_dir = './local_data/hek293'

    # Load config

    config_file = './local_data/configs/train-resnet-33.yaml'
    config = get_config(config_file)

    # Determine model ID

    model_id = model_file.split('.pth')[0].split('/')[-1]

    # Create test dataloader

    print("Creating test dataset...")

    test_cfile = f"{data_dir}/test_coding.pt"
    test_nfile = f"{data_dir}/test_noncoding.pt"
    test_data = SignalDataset(test_cfile, test_nfile)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # Get device for model evaluation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    model = ResNet(config.resnet).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)

    # Test

    model.eval()
    all_y_true = torch.tensor([])
    all_y_pred = torch.tensor([], device=device)
    all_y_pred_probs = torch.tensor([], device=device)
    batch_predict_t = []
    with torch.no_grad():
        for X, y in test_loader:

            # Move batch to GPU
            X = X.to(device)

            # Predict class probabilities
            start_t = time.time()
            y_pred_probs = softmax(model(X), dim=1)
            end_t = time.time()
            batch_predict_t.append(end_t-start_t)

            # Convert to class labels
            y_pred = torch.argmax(y_pred_probs, dim=1)

            # Store predictions
            all_y_true = torch.cat((all_y_true, y))
            all_y_pred = torch.cat((all_y_pred, y_pred))
            all_y_pred_probs = torch.cat((all_y_pred_probs, y_pred_probs))

    # Convert tensors to numpy arrays for downstream metrics

    all_y_true = all_y_true.numpy()
    all_y_pred = all_y_pred.cpu().numpy()
    all_y_pred_probs = all_y_pred_probs.cpu().numpy()

    # Compute accuracy

    n_correct = count_correct(all_y_true, all_y_pred)
    acc = n_correct / len(all_y_true) * 100
    print(f"Test accuracy: {acc:>0.1f}%\n")

    # Inference time

    max_t = max(batch_predict_t)
    min_t = min(batch_predict_t)
    avg_batch_t = mean(batch_predict_t)
    avg_pred_t = avg_batch_t / config.batch_size
    print(f"Max. batch inference time: {max_t}")
    print(f"Min. batch inference time: {min_t}")
    print(f"Avg. batch inference time: {avg_batch_t}")
    print(f"Avg. inference time per signal: {avg_pred_t}\n")

    # Compute confusion matrix

    matrix = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = matrix.ravel()
    print(f"Confusion matrix:\n------------------\n{matrix}")

    # Visualise confusion matrix

    categories = ['non-coding', 'coding'] # 0: non-coding, 1: coding
    sns.heatmap(matrix, annot=True, fmt='', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"{model_id} confusion matrix")
    plt.savefig(f"{model_id}_conf_matrix.png")
    plt.clf()

    ######################## TARGET: CODING ############################

    print("Metrics for target: CODING")

    # True positive rate (fraction of +ve class predicted correctly)

    cod_tpr = tp / (tp + fn)
    print(f"Coding TPR: {cod_tpr}")

    # False positive rate AKA recall (fraction of -ve class predicted incorrectly)

    cod_fpr = fp / (fp + tn)
    print(f"Coding FPR: {cod_fpr}")

    # Precision (fraction of correct +ve predictions)

    cod_prec = tp / (tp + fp)
    print(f"Coding Precision: {cod_prec}")

    # TP / FP rate

    cod_tp_fp = tp / fp
    print(f"Coding #TP/#FP: {cod_tp_fp}")

    # Compute ROC AUC

    coding_probs = all_y_pred_probs[:, 1]
    cod_auc = roc_auc_score(all_y_true, coding_probs)
    print(f"Coding AUC: {cod_auc:.3f}\n\n")

    # Plot ROC curve

    fpr, tpr, _ = roc_curve(all_y_true, coding_probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_id} coding ROC curve, AUC = {cod_auc:.3f}")
    plt.savefig(f"{model_id}_coding_roc_curve.png")


    ###################### TARGET: NON-CODING ##########################

    print("Metrics for target: NON-CODING")

    # True positive rate (fraction of +ve class predicted correctly)

    nc_tpr = tn / (tn + fp)
    print(f"Non-coding TPR: {nc_tpr}")

    # False positive rate AKA recall (fraction of -ve class predicted incorrectly)

    nc_fpr = fn / (fn + tp)
    print(f"Non-coding FPR: {nc_fpr}")

    # Precision (fraction of correct +ve predictions)

    nc_prec = tn / (tn + fn)
    print(f"Non-coding Precision: {nc_prec}")

    # TP / FP rate

    nc_tp_fp = tn / fn
    print(f"Non-coding #TP/#FP: {nc_tp_fp}")

    # Compute ROC AUC

    all_y_pred_probs_nc = []
    for probs in all_y_pred_probs:
        all_y_pred_probs_nc.append(list(probs[::-1]))
    all_y_pred_probs_nc = np.array(all_y_pred_probs_nc)

    all_y_true_nc = []
    for y in all_y_true:
        if y == 0:
            all_y_true_nc.append(1)
        else:
            all_y_true_nc.append(0)

    noncoding_probs = all_y_pred_probs_nc[:, 1]
    nc_auc = roc_auc_score(all_y_true_nc, noncoding_probs)
    print(f"Non-coding AUC: {nc_auc:.3f}\n\n")

    # Plot ROC curve

    fpr, tpr, _ = roc_curve(all_y_true_nc, noncoding_probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_id} non-coding ROC curve, AUC = {nc_auc:.3f}")
    plt.savefig(f"{model_id}_noncoding_roc_curve.png")

    print(f"{acc}/t{max_t}\t{min_t}\t{avg_batch_t}\t{avg_pred_t}\t{cod_tpr}\t{cod_fpr}\t{cod_prec}\t{cod_auc}\t{cod_tp_fp}\t{nc_tpr}\t{nc_fpr}\t{nc_prec}\t{nc_auc}\t{nc_tp_fp}")


if __name__ == "__main__":
    main()
 