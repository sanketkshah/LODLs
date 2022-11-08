from PThenO import PThenO
from SubmodularOptimizer import SubmodularOptimizer
from TopKOptimizer import TopKOptimizer

import torch
from torch.distributions import Normal, Bernoulli
import matplotlib.pyplot as plt
import pdb

from TopKOptimizer import TopKOptimizer


class RandomTopK(PThenO):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        budget=2,  # number of items that can be picked
        num_fake_targets=5000,  # number of random features added to make the task harder
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(RandomTopK, self).__init__()
        self._set_seed(rand_seed)

        # Generate Labels
        self.Ys_train, self.Ys_test = self._generate_labels(num_instances, num_items)  # labels

        # Generate features based on the data
        self.NUM_TARGETS = num_items
        self.num_fake_targets = num_fake_targets
        self.NUM_FEATURES = self.NUM_TARGETS + self.num_fake_targets
        self.Xs = self._generate_features(self.Ys, self.num_fake_targets)  # features
        assert not torch.isnan(self.Xs).any()

        # Split into train/testq
        assert 0 < test_frac < 1
        train_pct = (1 - test_frac) * 0.8   # Using 80% of non-test data for train, and 20% for validation
        self.test = range(0, int(test_frac * num_instances))
        self.train = range(int(test_frac * num_instances), int((train_pct + test_frac) * num_instances))
        self.val = range(int((train_pct + test_frac) * num_instances), num_instances)
        assert all(x is not None for x in [self.test, self.train, self.val])

        # Create functions for optimisation
        assert budget < num_items
        self.budget = budget
        self.opt = TopKOptimizer(self.get_objective, self.budget)

        # Undo random seed setting
        self._set_seed()

    def _generate_labels(
        self,
        num_instances, 
        num_items,
        prob_high=0.1,
        val_low=1.,
        val_high=10.,
        std=5.,
    ):
        """
        Loads the labels (Ys) of the prediction from the following distribution:
        y = N(3 (val_low), 1 (std)), with probability 0.9 (prob_high)
            N(10 (val_high), 1 (std)), with probability 0.1 (1 - prob_high)
        """

        # # Load N(val, std) for val_high and val_low
        # vals_high = Normal(torch.tensor(val_high), torch.tensor(std)).sample((num_instances, num_items))
        # vals_low = Normal(torch.tensor(val_low), torch.tensor(std)).sample((num_instances, num_items))

        # # Pick between them with probanility prob_high
        # choice = Bernoulli(torch.tensor(prob_high)).sample((num_instances, num_items))
        # Ys = choice * vals_high + (1 - choice) * vals_low

        Ys = Normal(torch.tensor(val_high), torch.tensor(std)).sample((num_instances, num_items))

        return Ys.float().detach()

    def _generate_features(self, Ys, num_fake_targets):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Normalise data
        Ys_standardised = (Ys - Ys.mean()) / (Ys.std() + 1e-10)
        assert not torch.isnan(Ys_standardised).any()

        # Add noise to the data to complicate prediction
        fake_features = torch.normal(mean=torch.zeros(Ys.shape[0], num_fake_targets))
        Ys_augmented = torch.cat((Ys_standardised, fake_features), dim=-1)

        # Encode Ys as features by multiplying them with a random matrix
        transform_nn = torch.nn.Sequential(torch.nn.Linear(Ys_augmented.shape[-1], Ys_augmented.shape[-1]))
        Xs = transform_nn(Ys_augmented).detach().clone()
        return Xs

    def get_train_data(self):
        return self.Xs[self.train], self.Ys[self.train], [None for _ in range(len(self.train))]

    def get_val_data(self):
        return self.Xs[self.val], self.Ys[self.val], [None for _ in range(len(self.val))]

    def get_test_data(self):
        return self.Xs[self.test], self.Ys[self.test], [None for _ in range(len(self.test))]

    def get_objective(
        self,
        Y,
        Z,
        alpha=1,
    ):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        obj = (Z * Y).sum(dim=-1) + alpha * Z.square().mean()
        return obj

    def get_decision(self, Y):
        # If this is a single instance of a decision problem
        if len(Y.shape) == 1:
            return self.opt(Y)

        # If it's not, break it down into individual instances and solve
        Z = torch.cat([self.opt(y).unsqueeze(0) for y in Y], dim=0)
        return Z


# Unit test for RandomTopK
if __name__ == '__main__':
    # Load An Example Instance
    problem = RandomTopK()

    # Plot It
    Ys = problem.Ys.flatten().tolist()
    plt.hist(Ys, bins=100)
    plt.show()
