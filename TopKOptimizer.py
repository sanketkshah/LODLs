import torch
import pdb
import numpy as np
from SubmodularOptimizer import OptimiseSubmodular


class TopKOptimizer(torch.nn.Module):
    """
    Wrapper around OptimiseSubmodular that saves state information.
    """
    def __init__(
        self,
        get_obj,  # A function that returns the value of the objective we want to minimise
        budget,  # The maximum number of items we can select (in expectation)
        lr=0.1,  # learning rate for optimiser
        momentum=0.9,  # momentum for optimiser
        num_iters=100,  # number of optimisation steps
        verbose=False  # print intermediate solution statistics
    ):
        super(TopKOptimizer, self).__init__()
        self.get_obj = get_obj
        self.budget = budget
        self.lr = lr
        self.momentum = momentum
        self.num_iters = num_iters
        self.verbose = verbose

    def forward(
        self,
        Yhat,  # predicted labels
    ):
        """
        Computes the optimal Z for the predicted Yhat using the supplied optimizer.
        """
        Z = OptimiseTopK.apply(Yhat, self.get_obj, self.budget, self.lr, self.momentum, self.num_iters, self.verbose)
        return Z


class OptimiseTopK(OptimiseSubmodular):
    """
    pytorch function for differentiable submodular maximization. The forward pass
    computes the optimal Z for a given set of predicted labels Yhat. The backward pass differentiates
    that optimal Z wrt Yhat.
    """
    @staticmethod
    def forward(
        ctx,
        Yhat,  # predicted labels
        get_obj,  # A function that returns the value of the objective we want to minimise
        budget,  # The maximum number of items we can select (in expectation)
        lr,  # learning rate for optimiser
        momentum,  # momentum for optimiser
        num_iters,  # number of optimisation steps
        verbose  # print intermediate solution statistics
    ):
        '''
        An optimised version of OptimiseSubmodular that works *only for vanilla top-k prediction*
        '''
        # Find the top "budget" items in Yhat
        _, Z_indices = Yhat.topk(budget, dim=-1)
        Z = torch.nn.functional.one_hot(Z_indices, Yhat.shape[-1]).sum(dim=-2).float()

        # Save things that are relevant to the backward pass
        ctx.save_for_backward(Yhat, Z)
        ctx.get_obj = get_obj
        ctx.budget = budget

        return Z

    @staticmethod
    def backward(ctx, grad_output):
        """
        Differentiates the optimal Z returned by the forward pass with respect
        to the ratings matrix that was given as input.
        """
        # Read saved tensor from forward pass
        Yhat, Z = ctx.saved_tensors
        get_obj = ctx.get_obj
        budget = ctx.budget

        # Compute the derivative of decision Z wrt the input predictions Yhat
        dZdYhat = OptimiseSubmodular._get_dZdYhat(Z, Yhat, get_obj, budget)

        # Apply chain rule
        dZdYhat_t = dZdYhat.t()
        out = torch.mm(dZdYhat_t.float(), grad_output.view(len(Z), 1))
        return out.view_as(Yhat), None, None, None, None, None, None  # Only Yhat gets a gradient

    @staticmethod
    def _get_elementwise_derivative(first_derivative, variable):
        second_derivative = []
        for element in first_derivative:
            second_derivative.append(torch.autograd.grad(element, variable, retain_graph=True)[0].unsqueeze(0))
        return torch.vstack(second_derivative)


    @staticmethod
    def _get_dZdYhat(
        Z,
        Yhat,
        get_obj,
        budget,
        EPS=1e-6,
        alpha=0.01,  # constant to be added to the diagonal of the A matrix to improve conditioning
    ):
        '''
        Returns the derivative of the optimal solution in the region around Z in
        terms of the predicted labels/rating matrix Yhat.

        Z: an optimal solution

        Yhat: the current parameter settings
        '''
        # Define useful constants
        num_var = len(Z)
        num_constr = 2 * num_var + 1

        # First, find the first and second order derivatives using autograd
        Yhat_local = Yhat.detach().requires_grad_(True)
        Z_local = Z.detach().requires_grad_(True)
        with torch.enable_grad():
            # Objective
            f = get_obj(Yhat_local, Z_local)
            # Gradient
            dfdZ = torch.autograd.grad(f, Z_local, create_graph=True)[0]
            # Hessian
            dfdZ_dZ = OptimiseSubmodular._get_elementwise_derivative(dfdZ, Z_local)
            # Cross-Term
            dfdZ_dYhat = OptimiseSubmodular._get_elementwise_derivative(dfdZ, Yhat_local)

        # Second, get the optimal dual variables via the KKT conditions
        #   dual variable for constraint sum(Z) <= k
        if torch.logical_and(Z > EPS, Z < 1 - EPS).any():
            lambda_sum = torch.mean(dfdZ[torch.logical_and(Z > EPS, Z < 1 - EPS)])
        else:
            lambda_sum = 0
        #   dual variable for constraint Z <= 1
        lambda_upper = torch.where(Z > 1 - EPS, dfdZ - lambda_sum, torch.zeros_like(Z))
        #   dual variable for constraint Z >= 0
        lambda_lower = torch.where(Z < EPS, dfdZ - lambda_sum, torch.zeros_like(Z))
        #   collect value of dual variables
        lam = torch.hstack((lambda_sum, lambda_upper, lambda_lower))
        diag_lambda = torch.diag(lam)

        # Third, collect value of constraints g(z) <= 0
        g_sum = Z_local.sum() - budget
        g_upper = Z_local - 1
        g_lower = -Z_local
        g = torch.hstack((g_sum, g_upper, g_lower))
        diag_g = torch.diag(g)

        # Fourth, get gradient of constraints wrt Z
        #   gradient of constraint sum(Z) <= k
        dgdZ_sum = torch.ones(num_var)
        #   gradient of constraints Z <= 1
        dgdZ_upper = torch.eye(num_var)
        #   gradient of constraints Z >= 0 <--> -Z <= 0
        dgdZ_lower = -torch.eye(num_var)
        dgdZ = torch.vstack((dgdZ_sum, dgdZ_upper, dgdZ_lower))

        # Putting it together, coefficient matrix for the linear system
        A = torch.vstack([torch.hstack([dfdZ_dZ, dgdZ.t()]), torch.hstack([diag_lambda @ dgdZ, diag_g])])
        # add alpha * I to improve conditioning
        A = A + alpha * torch.eye(num_var + num_constr)

        # RHS of the linear system, mostly partial derivative of grad f wrt Yhat
        b = torch.vstack([torch.hstack([dfdZ_dYhat.view((num_var, Yhat.numel()))]), torch.hstack([torch.zeros((num_constr, Yhat.numel()))])])

        # solution to the system
        derivatives = torch.linalg.solve(A, b)
        if torch.isnan(derivatives).any():
            print('report')
            print(torch.isnan(A).any())
            print(torch.isnan(b).any())
            print(torch.isnan(dgdZ).any())
            print(torch.isnan(diag_lambda).any())
            print(torch.isnan(diag_g).any())
            print(torch.isnan(dfdZ_dYhat).any())
            print(torch.isnan(dfdZ_dZ).any())

        # first num_var are derivatives of primal variables
        dZdYhat = derivatives[:num_var]
        return dZdYhat


# Unit test for submodular optimiser
if __name__ == '__main__':
    # Unit Test
    def get_obj(Y, Z):
        # Function to be *maximised*
        #   Sanity check inputs
        assert Y.shape[0] == Z.shape[0]
        assert len(Z.shape) == 1

        #   Compute submodular objective from Wilder, et. al. (2019)
        p_fail = 1 - Z.unsqueeze(1) * Y
        p_all_fail = p_fail.prod(dim=0)
        obj = (1 - p_all_fail).sum()
        return obj

    #   Load class
    opt = TopKOptimizer(get_obj, budget=1)

    #   Genereate data
    torch.manual_seed(100)
    Y = torch.rand((5, 10), requires_grad=True)

    #   Perform forward pass
    Z = opt(Y)
    loss = get_obj(Y, Z)
    print(loss)

    # Perform backward pass
    loss.backward()

    # Use torch.gradcheck to double check gradients
    torch.autograd.gradcheck(opt, Y)
