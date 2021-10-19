import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from resnet import ResNet, BottleneckBlock
from data import SignalDataset


def train(dataloader, model, loss_fn, optimizer, device):
    total = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Move data batch to GPU for propagation through network
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{total:>5d}]")


def validate(dataloader, model, loss_fn, device):
    total = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:

            # Move batch to GPU
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Compute average loss and accuracy
    val_loss /= n_batches
    correct /= total
    print(f"Validation error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return correct


def main():

    # TODO: CL args
    checkpt_dir = "/home/alex/Documents/rnaclassifier/saved_models"
    checkpt = None

    # Create datasets

    print("Creating datasets...")
    # data_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/5_MakeDataset"
    # train_cfile = f"{data_dir}/train_coding.pt"
    # train_nfile = f"{data_dir}/train_noncoding.pt"
    # valid_cfile = f"{data_dir}/val_coding.pt"
    # valid_nfile = f"{data_dir}/val_noncoding.pt"
    # batch_size = 1000

    data_dir = '/home/alex/Documents/rnaclassifier'
    train_cfile = f"{data_dir}/test_coding.pt"
    train_nfile = f"{data_dir}/test_noncoding.pt"
    valid_cfile = f"{data_dir}/test_coding.pt"
    valid_nfile = f"{data_dir}/test_noncoding.pt"
    batch_size = 64

    train_data = SignalDataset(train_cfile, train_nfile)
    valid_data = SignalDataset(valid_cfile, valid_nfile)

    # Create data loaders

    print("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # Get device for training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    model = ResNet(BottleneckBlock, [2,2,2,2]).to(device)
    if checkpt is not None:
        model.load_state_dict(torch.load(f"{checkpt_dir}/{checkpt}"))
    summary(model)

    # Define loss function & optimiser

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train

    n_epochs = 6
    best_acc = 0
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        val_acc = validate(valid_loader, model, loss_fn, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{checkpt_dir}/best-model.pth")

    print("Training complete.")



if __name__ == "__main__":
    main()
 