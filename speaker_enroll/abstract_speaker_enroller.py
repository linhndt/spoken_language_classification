from abc import ABC
from abc import abstractmethod
import pickle


class AbstractSpeakerEnroller(ABC):

    @abstractmethod
    def train(self, data_dict):
        pass

    @abstractmethod
    def get_label(self, data_file):
        pass


def load_from_file(file) -> AbstractSpeakerEnroller:
    return pickle.load(file)


def save_to_file(enroller: AbstractSpeakerEnroller, file):
    pickle.dump(enroller, file)