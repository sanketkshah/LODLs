from numpy import square
import torch
from math import sqrt
from functools import reduce
import operator
import pdb

from utils import View


# TODO: Pretty it up
def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

    if output_activation == 'relu':
        net_layers.append(torch.nn.ReLU())
    elif output_activation == 'sigmoid':
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == 'tanh':
        net_layers.append(torch.nn.Tanh())
    elif output_activation == 'softmax':
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)

class DenseLoss(torch.nn.Module):
    """
    A Neural Network-based loss function
    """

    def __init__(
        self,
        Y,
        num_layers=4,
        hidden_dim=100,
        activation='relu'
    ):
        super(DenseLoss, self).__init__()
        # Save true labels
        self.Y = Y.detach().view((-1))
        # Initialise model
        self.model = torch.nn.Parameter(dense_nn(Y.numel(), 1, num_layers, intermediate_size=hidden_dim, output_activation=activation))

    def forward(self, Yhats):
        # Flatten inputs
        Yhats = Yhats.view((-1, self.Y.numel()))

        return self.model(Yhats)


class WeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSE, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * self.weights.clamp(min=self.min_val)).mean(dim=-1)

        return weighted_mse


class WeightedMSEPlusPlus(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSEPlusPlus, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(min=self.min_val)
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(min=self.min_val)
        weights = pos_weights + neg_weights

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * weights).mean(dim=-1)

        return weighted_mse


class WeightedCE(torch.nn.Module):
    """
    A weighted version of CE
    """

    def __init__(self, Y, min_val=1):
        super(WeightedCE, self).__init__()
        # Save true labels
        self.Y_raw = Y.detach()
        self.Y = self.Y_raw.view((-1))
        self.num_dims = self.Y.shape[0]
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhat):
        # Flatten inputs
        if len(self.Y_raw.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.num_dims))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(min=self.min_val)
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(min=self.min_val)
        weights = pos_weights + neg_weights

        # Compute MSE
        error = torch.nn.BCELoss(reduction='none')(Yhat, self.Y.expand(*Yhat.shape))
        weighted_ce = (error * weights).mean(dim=-1)

        return weighted_ce


class WeightedMSESum(torch.nn.Module):
    """
    A weighted version of MSE-Sum
    """

    def __init__(self, Y):
        super(WeightedMSESum, self).__init__()
        # Save true labels
        assert len(Y.shape) == 2  # make sure it's a multi-dimensional input
        self.Y = Y.detach()

        # Initialise paramters
        self.msesum_weights = torch.nn.Parameter(torch.rand(Y.shape[0]))

    def forward(self, Yhats):
        # Get weighted MSE-Sum
        sum_error = (self.Y - Yhats).mean(dim=-1)
        row_error = sum_error.square()
        weighted_mse_sum = (row_error * self.msesum_weights).mean(dim=-1)

        return weighted_mse_sum


class TwoVarQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(self, Y):
        super(TwoVarQuadratic, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))

        # Initialise paramters
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, Yhat):
        """
        """
        # Flatten inputs
        Yhat = Yhat.view((Yhat.shape[0], -1))

        # Difference of squares
        # Gives diagonal elements
        diag = (self.Y - Yhat).square().mean()

        # Difference of sum of squares
        # Gives cross-terms
        cross = (self.Y - Yhat).mean().square()

        return self.alpha * diag + self.beta * cross


class QuadraticPlusPlus(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        quadalpha=1e-3,  # regularisation weight
        **kwargs
    ):
        super(QuadraticPlusPlus, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.num_dims = self.Y.shape[0]

        # Create quadratic matrices
        bases = torch.rand((self.num_dims, self.num_dims, 4)) / (self.num_dims * self.num_dims)
        self.bases = torch.nn.Parameter(bases)  

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.num_dims))

        # Measure distance between predicted and true distributions
        diff = (self.Y - Yhat).unsqueeze(-2)

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = self._get_basis(Yhat).clamp(-10, 10)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1).squeeze()

        return quad + self.alpha * mse

    def _get_basis(self, Yhats):
        # Figure out which entries to pick
        #   Are you above or below the true label
        direction = (Yhats > self.Y).type(torch.int64)
        #   Use this to figure out the corresponding index
        direction_col = direction.unsqueeze(-1)
        direction_row = direction.unsqueeze(-2)
        index = (direction_col + 2 * direction_row).unsqueeze(-1)

        # Pick corresponding entries
        bases = self.bases.expand(*Yhats.shape[:-1], *self.bases.shape)
        basis = bases.gather(-1, index).squeeze()
        return torch.tril(basis)

class LowRankQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        rank=2,  # rank of the learned matrix
        quadalpha=0.1,  # regularisation weight
        **kwargs
    ):
        super(LowRankQuadratic, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))

        # Create a quadratic matrix
        basis = torch.tril(torch.rand((self.Y.shape[0], rank)) / (self.Y.shape[0] * self.Y.shape[0]))
        self.basis = torch.nn.Parameter(basis)  

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.Y.shape[0]))

        # Measure distance between predicted and true distributions
        diff = self.Y - Yhat

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = torch.tril(self.basis).clamp(-100, 100)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1)

        return quad + self.alpha * mse


model_dict = {'dense': dense_nn}
