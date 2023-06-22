import sys
from statistics import mean
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchinfo import summary

from data import SignalDataset
from nets.cnn import ConvNet
from nets.resnet import ResNet
from nets.tcn import TCN
from utilities import get_config


def count_correct(y_true, y_pred):
    assert len(y_pred) == len(y_true)
    n_correct = 0
    for i, pred in enumerate(y_pred):
        if pred == y_true[i]:
            n_correct += 1
    return n_correct


def main():

    ############################### SETUP ##############################

    # CL args

    # model_file = sys.argv[1]
    # config_file = sys.argv[2]
    # data_dir = sys.argv[3]
    model_file = '/home/alex/Documents/tmp/globin/retrain-globin_0_best_model.pth'
    data_dir = '/home/alex/Documents/tmp/globin/globin-data'
    config_file = '/home/alex/Documents/tmp/globin/train-cnn-20-local.yaml'

    # Load config

    config = get_config(config_file)

    # Get info about experiment

    model_id = model_file.split('.pth')[0].split('/')[-1]
    arch = config_file.split('train-')[-1].split('-')[0]
    dataset = data_dir.split('/')[-1]
    print("##########################################################")
    print(f"\n\n\nTesting {arch} with id {model_id} on data {dataset}")
    print("##########################################################")

    # Create test dataloader

    print("Creating test dataset...")

    test_pfile = f"{data_dir}/test_positive.pt"
    test_nfile = f"{data_dir}/test_negative.pt"
    test_data = SignalDataset(test_pfile, test_nfile)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # Get device for model evaluation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    if arch == 'resnet':
        model = ResNet(config.resnet).to(device)
    elif arch == 'tcn':
        model = TCN(config.tcn).to(device)
    elif arch == 'cnn':
        model = ConvNet(config.cnn).to(device)
    else:
        print(f"Arch {arch} not defined!")
    model.load_state_dict(torch.load(model_file))
    summary(model)

    ########################## MODEL INFERENCE #########################

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

    ############################# ANALYSIS ############################

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
    print(f"Confusion matrix:\n------------------\n{matrix}\n")

    # Visualise confusion matrix

    categories = ['negative', 'positive'] # 0: negative, 1: positive
    sns.heatmap(matrix, annot=True, fmt='', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"{data_dir} {model_id} confusion matrix")
    plt.savefig(f"{data_dir}_{model_id}_conf_matrix.png")
    plt.clf()

    # True positive rate (fraction of +ve class predicted correctly)

    tpr = tp / (tp + fn)
    print(f"TPR: {tpr}")

    # False positive rate (fraction of -ve class predicted incorrectly)

    fpr = fp / (fp + tn)
    print(f"FPR: {fpr}")

    # Precision (fraction of correct +ve predictions)

    prec = tp / (tp + fp)
    print(f"Precision: {prec}")

    # TP / FP rate

    tp_fp = tp / fp
    print(f"#TP/#FP: {tp_fp}")

    # Compute ROC AUC

    positive_probs = all_y_pred_probs[:, 1]
    auc = roc_auc_score(all_y_true, positive_probs)
    print(f"AUC: {auc:.3f}\n")

    # Plot ROC curve

    fpr, tpr, _ = roc_curve(all_y_true, positive_probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{data_dir} {model_id} ROC curve, AUC = {auc:.3f}")
    plt.savefig(f"{data_dir}_{model_id}_roc_curve.png")


if __name__ == "__main__":
    main()
 