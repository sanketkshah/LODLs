from abc import ABC, abstractmethod
import random
import torch
import numpy as np
import time


class PThenO(ABC):
    """A class that defines an arbitrary predict-then-optimise problem."""

    def __init__(self):
        super(PThenO, self).__init__()

    @abstractmethod
    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_val_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_modelio_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def get_output_activation(self):
        raise NotImplementedError()

    @abstractmethod
    def get_twostageloss(self):
        raise NotImplementedError()

    @abstractmethod
    def get_decision(self, Y, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_objective(self, Y, Z, **kwargs):
        raise NotImplementedError()

    def _set_seed(self, rand_seed=int(time.time())):
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
