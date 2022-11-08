import random
import torch
import pdb
import os
import time
import pickle
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
from statistics import mean
from functools import partial
from copy import deepcopy

from models import DenseLoss, LowRankQuadratic, WeightedMSESum, WeightedMSE, WeightedCE, WeightedMSESum, QuadraticPlusPlus, WeightedMSEPlusPlus
from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from RMAB import RMAB
from utils import find_saved_problem, starmap_with_kwargs
NUM_CPUS = os.cpu_count()


def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()


def MAE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).abs().mean()


def CE(Yhats, Ys, **kwargs):
    return torch.nn.BCELoss()(Yhats, Ys)

def MSE_Sum(
    Yhats,
    Ys,
    alpha=0.1,  # weight of MSE-based regularisation
    **kwargs
):
    """
    Custom loss function that the squared error of the _sum_
    along the last dimension plus some regularisation.
    Useful for the Submodular Optimisation problems in Wilder et. al.
    """
    # Check if prediction is a matrix/tensor
    assert len(Ys.shape) >= 2

    # Calculate loss
    sum_loss = (Yhats - Ys).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(Yhats, Ys)
    return loss_regularised

def _sample_points(
    Y,  # The set of true labels
    problem,  # The optimisation problem at hand
    sampling,  # The method for sampling points
    num_samples,  # Number of points with which to fit model
    Y_aux=None,  # Extra information needed to solve the problem
    sampling_std=None,  # Standard deviation for the training data
    num_restarts=10,  # The number of times to run the optimisation problem for Z_opt
):
    # Sample points in the neighbourhood
    #   Find the rough scale of the predictions
    try:
        Y_std = float(sampling_std)
    except TypeError:
        Y_std = torch.std(Y) + 1e-5
    #   Generate points
    if sampling == 'random':
        #   Create some noise
        Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_uniform':
        #   Create some noise
        Y_noise = torch.distributions.Uniform(0, Y_std).sample((num_samples, *Y.shape))
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_dropout':
        #   Create some noise
        Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
        #   Drop some of the entries randomly
        drop_idxs = torch.distributions.Bernoulli(probs=0.1).sample((num_samples, *Y.shape))
        Y_noise = Y_noise * drop_idxs
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_flip':
        assert 0 < Y_std < 1
        #   Randomly choose some indices to flip
        flip_idxs = torch.distributions.Bernoulli(probs=Y_std).sample((num_samples, *Y.shape))
        #   Flip chosen indices to get sampled points
        Yhats = torch.logical_xor(Y, flip_idxs).float()
    elif sampling == 'numerical_jacobian':
        #   Find some points using this
        Yhats_plus = Y + (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
        Yhats_minus = Y - (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
        Yhats = torch.cat((Yhats_plus, Yhats_minus), dim=0)
    elif sampling == 'random_jacobian':
        #   Find dimensions to perturb and how much to perturb them by
        idxs = torch.randint(Y.numel(), size=(num_samples,))
        idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
        noise_scale = torch.distributions.Normal(0, Y_std).sample((num_samples,)).unsqueeze(dim=-1)
        noise = (idxs * noise_scale).view((num_samples, *Y.shape))
        #   Find some points using this
        Yhats = Y + noise
    elif sampling == 'random_hessian':
        #   Find dimensions to perturb and how much to perturb them by
        noise = torch.zeros((num_samples, *Y.shape))
        for _ in range(2):
            idxs = torch.randint(Y.numel(), size=(num_samples,))
            idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
            noise_scale = torch.distributions.Normal(0, Y_std).sample((num_samples,)).unsqueeze(dim=-1)
            noise += (idxs * noise_scale).view((num_samples, *Y.shape))
        #   Find some points using this
        Yhats = Y + noise
    else:
        raise LookupError()
    #   Make sure that the points are valid predictions
    if isinstance(problem, BudgetAllocation) or isinstance(problem, BipartiteMatching):
        Yhats = Yhats.clamp(min=0, max=1)  # Assuming Yhats must be in the range [0, 1]
    elif isinstance(problem, RMAB):
        Yhats /= Yhats.sum(-1, keepdim=True)

    # Calculate decision-focused loss for points
    opt = partial(problem.get_decision, isTrain=False, aux_data=Y_aux)
    obj = partial(problem.get_objective, aux_data=Y_aux)

    #   Calculate for 'true label'
    best = None
    assert num_restarts > 0
    for _ in range(num_restarts):
        Z_opt = opt(Y)
        opt_objective = obj(Y, Z_opt)

        if best is None or opt_objective > best[1]:
            best = (Z_opt, opt_objective)
    Z_opt, opt_objective = best

    #   Calculate for Yhats
    Zs = opt(Yhats, Z_init=Z_opt)
    objectives = obj(Y.unsqueeze(0).expand(*Yhats.shape), Zs)

    return (Y, opt_objective, Yhats, objectives)

def _learn_loss(
    problem,  # The problem domain
    dataset,  # The data set on which to train SL
    model_type,  # The model we're trying to fit
    num_iters=100,  # Number of iterations over which to train model
    lr=1,  # Learning rate with which to train the model
    verbose=False,  # print training loss?
    train_frac=0.3,  # fraction of samples to use for training
    val_frac=0.3,  # fraction of samples to use for testing
    val_freq=1,  # the number of training steps after which to check loss on val set
    print_freq=5,  # the number of val steps after which to print losses
    patience=25,  # number of iterations to wait for the train loss to improve when learning
    **kwargs
):
    """
    Function that learns a model to approximate the behaviour of the
    'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
    """
    # Get samples from dataset
    Y, opt_objective, Yhats, objectives = dataset
    objectives = opt_objective - objectives

    # Split train and test
    assert train_frac + val_frac < 1
    train_idxs = range(0, int(train_frac * Yhats.shape[0]))
    val_idxs = range(int(train_frac * Yhats.shape[0]), int((train_frac + val_frac) * Yhats.shape[0]))
    test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

    Yhats_train, objectives_train = Yhats[train_idxs], objectives[train_idxs]
    Yhats_val, objectives_val = Yhats[val_idxs], objectives[val_idxs]
    Yhats_test, objectives_test = Yhats[test_idxs], objectives[test_idxs]

    # Load a model
    if model_type == 'dense':
        model = DenseLoss(Y)
    elif model_type == 'quad':
        model = LowRankQuadratic(Y, **kwargs)
    elif model_type == 'weightedmse':
        model = WeightedMSE(Y)
    elif model_type == 'weightedmse++':
        model = WeightedMSEPlusPlus(Y)
    elif model_type == 'weightedce':
        model = WeightedCE(Y)
    elif model_type == 'weightedmsesum':
        model = WeightedMSESum(Y)
    elif model_type == 'quad++':
        model = QuadraticPlusPlus(Y, **kwargs)
    else:
        raise LookupError()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(device), Yhats_val.to(device), Yhats_test.to(device)
        objectives_train, objectives_val, objectives_test = objectives_train.to(device), objectives_val.to(device), objectives_test.to(device)
        model = model.to(device)

    # Fit a model to the points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), None)
    time_since_best = 0
    for iter_idx in range(num_iters):
        # Define update step using "closure" function
        def loss_closure():
            optimizer.zero_grad()
            pred = model(Yhats_train).flatten()
            if not (pred >= -1e-3).all().item():
                print(f"WARNING: Prediction value < 0: {pred.min()}")
            loss = MSE(pred, objectives_train)
            loss.backward()
            return loss

        # Perform validation
        if iter_idx % val_freq == 0:
            # Get performance on val dataset
            pred_val = model(Yhats_val).flatten()
            loss_val = MSE(pred_val, objectives_val)

            # Print statistics
            if verbose and iter_idx % (val_freq * print_freq) == 0:
                print(f"Iter {iter_idx}, Train Loss MSE: {loss_closure().item()}")
                print(f"Iter {iter_idx}, Val Loss MSE: {loss_val.item()}")
            # Save model if it's the best one
            if best[1] is None or loss_val.item() < best[0]:
                best = (loss_val.item(), deepcopy(model))
                time_since_best = 0
            # Stop if model hasn't improved for patience steps
            if time_since_best > patience:
                break

        # Make an update step
        optimizer.step(loss_closure)
        time_since_best += 1
    model = best[1]
    # If needed, PSDify
    # TODO: Figure out a better way to do this?
    # pdb.set_trace()
    # if hasattr(model, 'PSDify') and callable(model.PSDify):
    #     model.PSDify()

    # Get final loss on train samples
    pred_train = model(Yhats_train).flatten()
    train_loss = torch.nn.L1Loss()(pred_train, objectives_train).item()

    # Get loss on holdout samples
    pred_test = model(Yhats_test).flatten()
    loss = torch.nn.L1Loss()(pred_test, objectives_test)
    test_loss = loss.item()

    # Visualise generated datapoints and model
    # pdb.set_trace()
    # #   Visualise results on sampled_points
    # Yhats_flat = Yhats_train.reshape((Yhats_train.shape[0], -1))
    # Y_flat = Y.flatten()
    # Y_idx = random.randrange(Yhats_flat.shape[-1])
    # sample_idx = (Yhats_flat[:, Y_idx] - Y_flat[Y_idx]).square() > 0
    # pred = model(Yhats_train)
    # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), objectives_train[sample_idx].tolist(), label='true')
    # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), pred[sample_idx].tolist(), label='pred')
    # plt.legend(loc='upper right')
    # plt.show()

    # # Visualise results on random_direction
    # Y_flat = Y.flatten()
    # direction = 2 * torch.rand_like(Y_flat) - 1
    # direction = direction / direction.norm()
    # Y_range = 1
    # scale = torch.linspace(-Y_range, Y_range, 1000)
    # Yhats_flat = scale.unsqueeze(-1) * direction.unsqueeze(0) + Y_flat.unsqueeze(0)
    # pred = model(Yhats_flat)
    # true_dl = problem.get_objective(problem.get_decision(Yhats_flat), Yhats_flat) - problem.get_objective(problem.get_decision(Y_flat), Y_flat)
    # plt.scatter(scale.tolist(), true_dl.tolist(), label='true')
    # plt.scatter(scale.tolist(), pred.tolist(), label='pred')
    # plt.legend(loc='upper right')
    # plt.show()

    return model, train_loss, test_loss


def _get_learned_loss(
    problem,
    model_type='weightedmse',
    folder='models',
    num_samples=400,
    sampling='random',
    sampling_std=None,
    serial=True,
    **kwargs
):
    print("Learning Loss Functions...")

    # Learn Losses
    #   Get Ys
    _, Y_train, Y_train_aux = problem.get_train_data()
    _, Y_val, Y_val_aux = problem.get_val_data()

    #   Get points in the neighbourhood of the Ys
    #       Try to load sampled points
    master_filename = os.path.join(folder, f"{problem.__class__.__name__}.csv")
    problem_filename, _ = find_saved_problem(master_filename, problem.__dict__)
    samples_filename_read = f"{problem_filename[:-4]}_{sampling}_{sampling_std}.pkl"

    # Check if there are enough stored samples
    num_samples_needed = num_extra_samples = num_samples
    if os.path.exists(samples_filename_read):
        with open(samples_filename_read, 'rb') as filehandle:
            num_existing_samples, SL_dataset_old = pickle.load(filehandle)
    else:
        num_existing_samples = 0
        SL_dataset_old = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}

    # Sample more points if needed
    num_samples_needed = num_samples
    num_extra_samples = max(num_samples_needed - num_existing_samples, 0)
    datasets = [entry for entry in zip([Y_train, Y_val], [Y_train_aux, Y_val_aux], ['train', 'val'])]
    if num_extra_samples > 0:
        SL_dataset = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}
        for Ys, Ys_aux, partition in datasets:
            # Get new sampled points
            start_time = time.time()
            if serial == True:
                sampled_points = [_sample_points(Y, problem, sampling, num_extra_samples, Y_aux, sampling_std) for Y, Y_aux in zip(Ys, Ys_aux)]
            else:
                with Pool(NUM_CPUS) as pool:
                    sampled_points = pool.starmap(_sample_points, [(Y, problem, sampling, num_extra_samples, Y_aux, sampling_std) for Y, Y_aux in zip(Ys, Ys_aux)])
            print(f"({partition}) Time taken to generate {num_extra_samples} samples for {len(Ys)} instances: {time.time() - start_time}")

            # Use them to augment existing sampled points
            for idx, (Y, opt_objective, Yhats, objectives) in enumerate(sampled_points):
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)

        # Save dataset
        samples_filename_write = f"{problem_filename[:-4]}_{sampling}_{sampling_std}_{time.time()}.pkl"
        with open(samples_filename_write, 'wb') as filehandle:
            pickle.dump((num_extra_samples, SL_dataset), filehandle)

        #   Augment with new data
        for Ys, Ys_aux, partition in datasets:
            for idx, Y in enumerate(Ys):
                # Get old samples
                Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[partition][idx]
                Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[partition][idx]
                assert torch.isclose(Y_old, Y).all()
                assert torch.isclose(Y_new, Y).all()

                # Combine entries
                opt_objective = opt_objective_new if opt_objective_old is None else max(opt_objective_new, opt_objective_old)
                Yhats = Yhats_new if Yhats_old is None else torch.cat((Yhats_old, Yhats_new), dim=0)
                objectives = objectives_new if objectives_old is None else torch.cat((objectives_old, objectives_new), dim=0)

                # Update
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)
        num_existing_samples += num_extra_samples
    else:
        SL_dataset = SL_dataset_old

    #   Learn SL based on the sampled Yhats
    train_maes, test_maes, avg_dls = [], [], []
    losses = {}
    for Ys, Ys_aux, partition in datasets:
        # Sanity check that the saved data is the same as the problem's data
        for idx, (Y, Y_aux) in enumerate(zip(Ys, Ys_aux)):
            Y_dataset, opt_objective, _, objectives = SL_dataset[partition][idx]
            assert torch.isclose(Y, Y_dataset).all()

            # Also log the "average error"
            avg_dls.append((opt_objective - objectives).abs().mean().item())

        # Get num_samples_needed points
        random.seed(0)  # TODO: Remove. Temporary hack for reproducibility.
        idxs = random.sample(range(num_existing_samples), num_samples_needed)
        random.seed()

        # Learn a loss
        start_time = time.time()
        if serial == True:
            losses_and_stats = [_learn_loss(problem, (Y_dataset, opt_objective, Yhats[idxs], objectives[idxs]), model_type, **kwargs) for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[partition]]
        else:
            with Pool(NUM_CPUS) as pool:
                losses_and_stats = starmap_with_kwargs(pool, _learn_loss, [(problem, (Y_dataset, opt_objective.detach().clone(), Yhats[idxs].detach().clone(), objectives[idxs].detach().clone()), deepcopy(model_type)) for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[partition]], kwargs=kwargs)
        print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {time.time() - start_time}")

        # Parse and log results
        losses[partition] = []
        for learned_loss, train_mae, test_mae in losses_and_stats:
            train_maes.append(train_mae)
            test_maes.append(test_mae)
            losses[partition].append(learned_loss)

    # Print overall statistics
    print(f"\nMean Train DL - OPT: {mean(avg_dls)}")
    print(f"Train MAE for SL: {mean(train_maes)}")
    print(f"Test MAE for SL: {mean(test_maes)}\n")

    # Return the loss function in the expected form
    def surrogate_decision_quality(Yhats, Ys, partition, index, **kwargs):
        return losses[partition][index](Yhats).flatten() - SL_dataset[partition][index][1]
    return surrogate_decision_quality


def _get_decision_focused(
    problem,
    dflalpha=1.,
    **kwargs,
):
    if problem.get_twostageloss() == 'mse':
        twostageloss = MSE
    elif problem.get_twostageloss() == 'ce':
        twostageloss = CE
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")

    def decision_focused_loss(Yhats, Ys, **kwargs):
        Zs = problem.get_decision(Yhats, isTrain=True, **kwargs)
        obj = problem.get_objective(Ys, Zs, isTrain=True, **kwargs)
        loss = -obj + dflalpha * twostageloss(Yhats, Ys)

        return loss

    return decision_focused_loss


def get_loss_fn(
    name,
    problem,
    **kwargs
):
    if name == 'mse':
        return MSE
    elif name == 'msesum':
        return MSE_Sum
    elif name == 'ce':
        return CE
    elif name == 'dfl':
        return _get_decision_focused(problem, **kwargs)
    else:
        return _get_learned_loss(problem, name, **kwargs)
