import argparse
import torch
import random
import pdb
import matplotlib.pyplot as plt

from RandomTopK import RandomTopK
from models import model_dict
from losses import get_loss_fn


if __name__ == '__main__':
    # Get hyperparams from the command line
    # TODO: Do this for all the domains together
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--testfrac', type=float, default=0.5)
    parser.add_argument('--instances', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['dense'], default='dense')
    parser.add_argument('--loss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmsesum', 'dfl', 'quad'], default='mse')
    parser.add_argument('--fakefeatures', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--sampling', type=str, choices=['random', 'numerical_jacobian', 'random_jacobian'], default='random_jacobian')
    #   Domain-specific
    parser.add_argument('--numitems', type=int, default=100)
    parser.add_argument('--budget', type=int, default=2)
    args = parser.parse_args()

    # Load problem
    print("Loading Problem...")
    problem = RandomTopK(
        num_instances=args.instances,
        test_frac=args.testfrac,
        rand_seed=args.seed,
        num_fake_targets=args.fakefeatures,
        num_items=args.numitems,
        budget=args.budget,
    )

    # Load a loss function to train the ML model on
    #   TODO: Abstract over this loss for the proposed method
    #   TODO: Figure out loss function "type" for mypy type checking
    print("Loading Loss Function...")
    loss_fn = get_loss_fn(args.loss, problem)

    # Load an ML model to predict the parameters of the problem
    #   TODO: Abstract over models? What should model builder look like in general?
    print("Building Model...")
    model_builder = model_dict[args.model]
    model = model_builder(
        problem.NUM_FEATURES,
        problem.NUM_TARGETS,
        args.layers,
        output_activation=None,
    )
    #   TODO: Add ways to modify lr, etc?
    optimizer = torch.optim.Adam(model.parameters())

    # Train neural network with a given loss function
    #   TODO: Add early stopping
    print(f"Training {args.model} model on {args.loss} loss...")
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    for iter_idx in range(args.iters):
        losses = []
        for i in random.sample(range(len(problem.train)), min(args.batchsize, len(problem.train))):
            pred = model(X_train[i])
            losses.append(loss_fn(pred, Y_train[i], index=i))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print metrics
        if iter_idx % 5 == 0:
            X_val, Y_val = problem.get_val_data()
            pred = model(X_val)
            Z_val = problem.get_decision(pred)
            objectives = problem.get_objective(Y_val, Z_val)
            print(f"Iter {iter_idx}, Train Loss: {loss.item()}, Val Decision Quality: {objectives.mean().item()}")
    #   TODO: Save the learned model

    # Document how well this trained model does
    print("Benchmarking Model...")
    X_test, Y_test = problem.get_test_data()
    pred = model(X_test)

    #   Plot predictions on test data
    plt.hist(pred.flatten().tolist(), bins=100, alpha=0.5, label='pred')
    plt.hist(Y_test.flatten().tolist(), bins=100, alpha=0.5, label='true')
    plt.legend(loc='upper right')
    plt.show()

    #   In terms of loss function
    if args.loss in ["mse", "msesum"]:
        loss = loss_fn(pred, Y_test)
        print(f"\nTest Loss on {args.loss} loss: {loss.item()}")

    #   In terms of problem objective
    Z_test = problem.get_decision(pred)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"\nTest Decision Quality: {objectives.mean().item()}")

    #   Document the value of a random guess
    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")

    #   Document the optimal value
    Z_test = problem.get_decision(Y_test)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Optimal Decision Quality: {objectives.mean().item()}")
    print()
