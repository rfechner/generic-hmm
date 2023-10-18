from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode_series(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode_series(self, *args, **kwargs):
        pass
