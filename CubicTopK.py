from functools import partial
from unicodedata import decimal
from PThenO import PThenO
from RMABSolver import TopK_custom

import torch
from torch.distributions import Normal, Bernoulli
import random
import pdb


class CubicTopK(PThenO):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=100,  # number of instances to use from the dataset to test
        num_items=50,  # number of targets to consider
        budget=2,  # number of items that can be picked
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(CubicTopK, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)

        # Generate Dataset
        #   Save relevant parameters
        self.num_items = num_items
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        #   Generate features
        self._set_seed(train_seed)
        self.Xs_train = 2 * torch.rand(self.num_train_instances, self.num_items, 1) - 1
        self._set_seed(test_seed)
        self.Xs_test = 2 * torch.rand(self.num_test_instances, self.num_items, 1) - 1
        #   Generate Labels
        self.Ys_train = 10 * (self.Xs_train.pow(3) - 0.65 * self.Xs_train).squeeze()
        self.Ys_test = 10 * (self.Xs_test.pow(3) - 0.65 * self.Xs_test).squeeze()

        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Save variables for optimisation
        assert budget < num_items
        self.budget = budget

        # Undo random seed setting
        self._set_seed()

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs], [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs], [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, [None for _ in range(len(self.Ys_test))]

    def get_objective(self, Y, Z, **kwargs):
        return (Z * Y).sum(dim=-1)

    def opt_train(self, Y):
        gamma = TopK_custom(self.budget)(-Y).squeeze()
        Z = gamma[...,0] * Y.shape[-1]
        return Z
    
    def opt_test(self, Y):
        _, idxs = torch.topk(Y, self.budget)
        Z = torch.nn.functional.one_hot(idxs, Y.shape[-1])
        return Z if self.budget == 0 else Z.sum(dim=-2)

    def get_decision(self, Y, isTrain=False, **kwargs):
        return self.opt_train(Y) if isTrain else self.opt_test(Y)
    
    def get_modelio_shape(self):
        return 1, 1

    def get_output_activation(self):
        return None
    
    def get_twostageloss(self):
        return 'mse'


# Unit test for RandomTopK
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pdb

    # Load An Example Instance
    pdb.set_trace()
    problem = CubicTopK()

    # Plot It
    Xs = problem.Xs_train.flatten().tolist()
    Ys = problem.Ys_train.flatten().tolist()
    plt.scatter(Xs, Ys)
    plt.show()
