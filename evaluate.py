import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from resnet import ResNet, BottleneckBlock
from data import SignalDataset


def count_correct(y_true, y_pred):
    assert len(y_pred) == len(y_true)
    n_correct = 0
    for i, pred in enumerate(y_pred):
        if pred == y_true[i]:
            n_correct += 1
    return n_correct


def main():

    # CL args

    model_file = sys.argv[1]
    data_dir = sys.argv[2]
    batch_size = int(sys.argv[3])

    # Determine model ID

    model_id = model_file.split('.pth')[0].split('/')[-1]

    # Create test dataloader

    print("Creating test dataset...")

    test_cfile = f"{data_dir}/val_coding.pt"
    test_nfile = f"{data_dir}/val_noncoding.pt"
    test_data = SignalDataset(test_cfile, test_nfile)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Get device for model evaluation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    model = ResNet(BottleneckBlock, [2,2,2,2]).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)

    # Test

    model.eval()
    all_y_true = torch.tensor([])
    all_y_pred = torch.tensor([], device=device)
    all_y_pred_probs = torch.tensor([], device=device)
    with torch.no_grad():
        for X, y in test_loader:

            # Move batch to GPU
            X = X.to(device)

            # Predict class probabilities
            y_pred_probs = model(X) 
            # TODO: Check if there are errors downstream - these are the
            # raw logits, not the softmax probs.

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

    # Compute confusion matrix

    matrix = confusion_matrix(all_y_true, all_y_pred)
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

    # Compute ROC AUC

    coding_probs = all_y_pred_probs[:, 1]
    auc = roc_auc_score(all_y_true, coding_probs)
    print(f"AUC: {auc:.3f}")

    # Plot ROC curve

    fpr, tpr, _ = roc_curve(all_y_true, coding_probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_id} ROC curve, AUC = {auc:.3f}")
    plt.savefig(f"{model_id}_roc_curve.png")


if __name__ == "__main__":
    main()
 