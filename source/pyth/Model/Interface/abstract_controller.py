from abc import ABC, abstractmethod

class AbstractController(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot_anterior_distribution(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimal_state_sequence(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot_posterior_distribution(self, *args, **kwargs):
        pass

    @abstractmethod
    def construct(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def kfold_cross_validation(self, *args, **kwargs):
        pass
