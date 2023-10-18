from abc import ABC, abstractmethod
from source.pyth.Preprocessing.Interface.abstract_preprocessor import AbstractPreprocessor

class AbstractFeatureExtractor(ABC):

    def __init__(self, prep : AbstractPreprocessor, *args, **kwargs):
        pass

    @abstractmethod
    def extract_probabilities(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_layerinfo_from_layers(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_dateid_if_present(self, *args, **kwargs):
        pass