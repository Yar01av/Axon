from abc import ABC, abstractmethod


class Model(ABC):
    """
    A wrapper around different kinds of models with controllable learning rates.
    The interface is entirely in numpy.
    """

    @abstractmethod
    def __init__(self, model, lr=0.001, **kwargs):
        """
        :param model: Structure of the model (layers) excluding anything else
        :param lr: learning rate of the model
        """
        self.lr = lr
        self._model = model

    @abstractmethod
    def save(self, path):
        """
        Save the model to long-term storage
        :param path: where to save
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict on X
        :param X: list or ndarray to predict on
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, load_path):
        """
        Initializes the model based on the saved file
        :param load_path: where the model is saved
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Trains the model
        :type kwargs: some additional arguments
        :param X: input
        :param y: correct output
        :return: None
        """
        pass

    def get_model(self):
        return self._model
