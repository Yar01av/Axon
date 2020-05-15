from keras.engine.saving import load_model
from keras.optimizers import Adam

from models.model import Model


class KerasModel(Model):
    def __init__(self, model, lr=0.001, **kwargs):
        super().__init__(model, lr, **kwargs)

        model.compile(loss='mse',
                      optimizer=Adam(lr=lr))

    def save(self, path):
        self._model.save(path)

    def predict(self, X):
        return self._model.predict(X)

    def load_model(self, load_path):
        self._model = load_model(load_path)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y, batch_size=kwargs.get("batch_size", len(X)), epochs=1, verbose=0)
