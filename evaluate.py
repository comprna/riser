import cProfile

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from resnet import ResNet, BottleneckBlock
from data import SignalDataset

def callback():

    # TODO: CL args
    checkpt = "best-model.pth"

    # Create test dataloader

    # print("Creating datasets...")
    # checkpt_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/experiments/train-1"
    # data_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/5_MakeDataset"
    # test_cfile = f"{data_dir}/val_coding.pt"
    # test_nfile = f"{data_dir}/val_noncoding.pt"
    # batch_size = 1000

    checkpt_dir = "/home/alex/Documents/rnaclassifier/saved_models"
    data_dir = '/home/alex/Documents/rnaclassifier/local_data'
    test_cfile = f"{data_dir}/test_coding.pt"
    test_nfile = f"{data_dir}/test_noncoding.pt"
    batch_size = 64

    test_data = SignalDataset(test_cfile, test_nfile)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Get device for training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    model = ResNet(BottleneckBlock, [2,2,2,2]).to(device)
    model.load_state_dict(torch.load(f"{checkpt_dir}/{checkpt}"))
    summary(model)

    # Test

    model.eval()
    all_y_pred = []
    all_y_true = []
    with torch.no_grad():
        for X, y in test_loader:

            # Move batch to GPU
            X = X.to(device)

            # Predict class probabilities
            y_pred = model(X)

            # Convert to class labels
            y_pred = torch.argmax(y_pred, dim=1)

            # Return to CPU for remaining operations
            y_pred = y_pred.cpu().numpy().tolist()
            y = y.numpy().tolist()

            # Store predictions
            all_y_pred.extend(y_pred)
            all_y_true.extend(y)

    # Compute accuracy

    n_correct = 0
    for i, pred in enumerate(all_y_pred):
        if pred == all_y_true[i]:
            n_correct += 1
    acc = n_correct / len(all_y_true) * 100
    print(f"Test accuracy: {acc:>0.1f}%\n")

    # Compute confusion matrix

    matrix = confusion_matrix(all_y_true, all_y_pred)
    print(f"Confusion matrix:\n------------------\n{matrix}")

    # Visualise confusion matrix

    categories = ['coding', 'non-coding'] # 0: coding, 1: non-coding
    sns.heatmap(matrix, annot=True, fmt='', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()



# def count_correct(y_true, y_pred):
#     assert len(y_pred) == len(y_true)
#     n_correct = 0
#     for i, pred in enumerate(y_pred):
#         if pred == y_true[i]:
#             n_correct += 1
#     return n_correct


def main():

    cProfile.run('callback()', sort='cumtime')

    # # TODO: CL args
    # checkpt = "best-model.pth"

    # # Create test dataloader

    # # print("Creating datasets...")
    # # checkpt_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/experiments/train-1"
    # # data_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/5_MakeDataset"
    # # test_cfile = f"{data_dir}/val_coding.pt"
    # # test_nfile = f"{data_dir}/val_noncoding.pt"
    # # batch_size = 1000

    # checkpt_dir = "/home/alex/Documents/rnaclassifier/saved_models"
    # data_dir = '/home/alex/Documents/rnaclassifier/local_data'
    # test_cfile = f"{data_dir}/test_coding.pt"
    # test_nfile = f"{data_dir}/test_noncoding.pt"
    # batch_size = 64

    # test_data = SignalDataset(test_cfile, test_nfile)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # # Get device for training

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device)
    # print(f"Using {device} device")

    # # Define model

    # model = ResNet(BottleneckBlock, [2,2,2,2]).to(device)
    # model.load_state_dict(torch.load(f"{checkpt_dir}/{checkpt}"))
    # summary(model)

    # # Test

    # model.eval()
    # all_y_pred = torch.tensor([]).to(device)
    # all_y_true = torch.tensor([]).to(device)
    # with torch.no_grad():
    #     for X, y in test_loader:

    #         # Move batch to GPU
    #         X, y = X.to(device), y.to(device)

    #         # Predict class probabilities
    #         y_pred = model(X)

    #         # Convert to class labels
    #         y_pred = torch.argmax(y_pred, dim=1)

    #         # Store predictions
    #         all_y_pred = torch.cat((all_y_pred, y_pred))
    #         all_y_true = torch.cat((all_y_true, y))

    #         # TODO: Is it faster to do all this on CPU with lists???

    # # Compute accuracy

    # n_correct = count_correct(all_y_true, all_y_pred)
    # acc = n_correct / len(all_y_true) * 100
    # print(f"Test accuracy: {acc:>0.1f}%\n")

    # # Compute confusion matrix

    # all_y_true = all_y_true.cpu().numpy()
    # all_y_pred = all_y_pred.cpu().numpy()
    # matrix = confusion_matrix(all_y_true, all_y_pred)
    # print(f"Confusion matrix:\n------------------\n{matrix}")

    # # Visualise confusion matrix

    # categories = ['coding', 'non-coding'] # 0: coding, 1: non-coding
    # sns.heatmap(matrix, annot=True, fmt='', cmap='Blues', 
    #             xticklabels=categories, yticklabels=categories)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()





if __name__ == "__main__":
    main()
 