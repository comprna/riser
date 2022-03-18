import torch
from torchinfo import summary

from cnn import ConvNet


class Model():
    def __init__(self, saved_model, config):
        self.device = self._setup_device()
        self.model = ConvNet(config.cnn).to(self.device)
        self.model.load_state_dict(torch.load(saved_model))
        summary(self.model)
        self.model.eval()

    def classify(self, signal):
        with torch.no_grad():
            X = torch.from_numpy(signal).unsqueeze(0)
            X = X.to(self.device, dtype=torch.float)
            logits = self.model(X)
            return torch.argmax(logits, dim=1)

    def _setup_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using {device} device")
        return device
