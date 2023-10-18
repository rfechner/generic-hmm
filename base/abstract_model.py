from abc import ABC, abstractmethod
from base.abstract_feature_extractor import AbstractFeatureExtractor

class AbstractModel(ABC):

    def __init__(self, fe : AbstractFeatureExtractor):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def iter_approx_stationary_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def posterior_distribution(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass