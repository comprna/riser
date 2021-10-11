import torch
from torch import nn
from torch.utils.data import DataLoader

from resnet import ResNet, BottleneckBlock
from data import SignalDataset

def train(dataloader, model, loss_fn, optimizer, device):
    total = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Move data batch to GPU for propagation through network
        X, y = X.to(device), y.to(device)

        print(type(X))
        print(type(y))
        exit()

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
    with torch.no_grad:
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


def main():

    # Create datasets

    print("Creating datasets...")
    data_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/5_MakeDataset"
    train_cfile = f"{data_dir}/train_coding.pt"
    train_nfile = f"{data_dir}/train_noncoding.pt"
    valid_cfile = f"{data_dir}/val_coding.pt"
    valid_nfile = f"{data_dir}/val_noncoding.pt"

    train_data = SignalDataset(train_cfile, train_nfile)
    valid_data = SignalDataset(valid_cfile, valid_nfile)

    # Create data loaders

    print("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1000, shuffle=False)

    # Get device for training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

    model = ResNet(BottleneckBlock, [2,2,2,2])
    print(f"Model: \n{model}")

    # Define loss function & optimiser

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train

    n_epochs = 6
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        validate(valid_loader, model, loss_fn, device)
    print("Training complete.")



if __name__ == "__main__":
    main()
 