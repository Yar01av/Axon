from warnings import warn

from torch import nn, optim, tensor, float32
from models.model import Model


class TorchModel(Model):
    def __init__(self, model, lr=0.001):
        model.float().to("cuda")
        super().__init__(model, lr)

        self._loss_function = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

    def save(self, path):
        warn("Saving not implemented")

    def predict(self, X):
        return self._model(tensor(X, dtype=float32)
                   .to("cuda"))\
                   .detach() \
                   .to("cpu") \
                   .numpy()

    def load_model(self, load_path):
        warn("Loading not implemented")

    def fit(self, X, y, **kwargs):
        predictions = self._model(tensor(X, dtype=float32).to("cuda"))
        loss = self._loss_function(predictions, tensor(y, dtype=float32).to("cuda"))
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
