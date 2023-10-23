import torch

from nets.cnn import ConvNet


class Model():
    def __init__(self, state, config, logger, target):
        self.target = target

        # Logger
        self.logger = logger # TODO: Make private?? How to deal with passing logger around??

        # Device to run model on
        self.device = self._get_device()
        self.logger.info('Using %s device', self.device)

        # Build CNN for testing
        self.model = ConvNet(config.cnn).to(self.device)
        self.model.load_state_dict(torch.load(state))
        self.model.eval()

    def classify(self, signal):
        with torch.no_grad():
            X = torch.from_numpy(signal).unsqueeze(0)
            X = X.to(self.device, dtype=torch.float)
            logits = self.model(X)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        return probs

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
