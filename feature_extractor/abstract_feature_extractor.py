from abc import ABC, abstractmethod


class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, rate, signal):
        pass

    @abstractmethod
    def get_data(self):
        pass