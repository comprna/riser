import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from resnet import ResNet, BottleneckBlock
from data import SignalDataset


def train(dataloader, model, loss_fn, optimizer, device, writer, epoch, log_freq=100):
    model.train()
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):

        # Move data batch to GPU for propagation through network
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch != 0 and batch % log_freq == 0:
            sample = batch * len(X)
            global_step = epoch * n_samples + sample
            avg_loss = total_loss / batch
            print(f"loss: {avg_loss:>7f} [{sample:>5d}/{n_samples:>5d}]")
            writer.add_scalar('training loss', avg_loss, global_step)

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(dataloader, model, loss_fn, device, writer, epoch):
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    total_loss, n_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:

            # Move batch to GPU
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            n_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Compute average loss and accuracy
    avg_loss = total_loss / n_batches
    acc = n_correct / n_samples * 100
    print(f"Validation set: \n Accuracy: {acc:>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    writer.add_scalar('validation loss', avg_loss, epoch)
    writer.add_scalar('validation acc', acc, epoch)

    return avg_loss, acc


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

    # Set up TensorBoard

    writer = SummaryWriter() # TODO: runs/experiment_id

    # Train

    n_epochs = 6
    best_acc = 0
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device, writer, t)
        val_acc = validate(valid_loader, model, loss_fn, device, writer, t)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{checkpt_dir}/best-model.pth")
            print(f"Saved model at epoch {t+1}.")

    print("Training complete.")



if __name__ == "__main__":
    main()
 