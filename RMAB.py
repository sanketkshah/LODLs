from pip import main
from PThenO import PThenO
from itertools import product
import random
import pdb
import numpy as np
from RMABSolver import RMABSolver
import torch
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from models import dense_nn
from utils import gather_incomplete_left


# TODO: Remove default
class RMAB(PThenO):
    """A 2-state 2-action RMAB problem cobbled together from multiple sources."""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_arms=5,  # number of arms to consider
        eval_method='sim',  # evaluation method
        min_lift=0.2,  # minimum amount of benefit associated with acting
        budget=1,  # number of arms that can be picked per timestep
        gamma=0.99,  # discount factor
        num_features=16,  # number of features for the PThenO problem
        num_intermediate=64,  # number of intermediate nodes in the "scrambling" NN to generate features
        num_layers=3,  # number of layers in the "scrambling" NN
        noise_std=1,  # noise to be added to the features after scrambling
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(RMAB, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)

        # Generate rewards
        # TODO: Generalise to more than 2 states
        self.num_states = 2
        self.num_actions = 2
        R = np.arange(self.num_states)
        self.R = (R - np.min(R)) / np.ptp(R) # normalize rewards

        # Load train and test transition probabilities
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.min_lift = min_lift
        Ys_train_test = []
        for seed, num_instances in zip([train_seed, test_seed], [num_train_instances, num_test_instances]):
            # Set seed for reproducibility
            self._set_seed(seed)

            # Load the relevant data (Ys)
            Ys = self._generate_instances(self.R, num_instances, num_arms, self.num_states, self.num_actions, min_lift)  # labels
            assert not torch.isnan(Ys).any()

            # Save Xs and Ys
            Ys_train_test.append(Ys)
        self.Ys_train, self.Ys_test = (*Ys_train_test,)

        # Generate features based on the labels
        self.num_arms = num_arms
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_intermediate = num_intermediate
        self.noise_std = noise_std
        self.Xs_train, self.Xs_test = self._generate_features([self.Ys_train, self.Ys_test], self.num_layers, self.num_intermediate, self.num_features, self.noise_std)  # features
        assert not (torch.isnan(self.Xs_train).any() or torch.isnan(self.Xs_test).any())

        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Create functions for optimisation
        assert budget < num_arms
        self.gamma = gamma
        self.budget = budget
        self.opt_train = RMABSolver(budget, isTrain=True)
        self.opt_test = RMABSolver(budget, isTrain=False)
        self.eval_method = eval_method

        # Undo random seed setting
        self._set_seed()

    def _get_rand_distr(self, num_states):
        random_points = np.random.uniform(size=num_states-1)
        random_points = np.append(random_points, [0,1])
        sorted_points = sorted(random_points)
        diffs = np.diff(sorted_points)
        assert np.sum(diffs) == 1 and len(diffs) == num_states
        return diffs

    def _generate_instances(self, R, num_instances, num_arms, num_states, num_actions, min_lift):
        """
        From https://github.com/armman-projects/Google-AI/

        This function generates a num_arms x num_states x num_actions x num_states T matrix indexed as: \
        T[arm][current_state][action][next_state]
        action=0 denotes passive action, a=1 is active action
        """
        # Generate random transition probabilities          
        T = np.zeros((num_instances, num_arms, num_states, num_actions, num_states))
        for i in range(num_instances): 
            for j in range(num_arms):
                for k in range(num_states):
                    while True:
                        passive_transition = self._get_rand_distr(num_states)
                        active_transition  = self._get_rand_distr(num_states)
                        if active_transition @ R > passive_transition @ R + min_lift: # Ensure that calling is significantly better
                            T[i,j,k,0,:] = passive_transition
                            T[i,j,k,1,:] = active_transition
                            break

        return torch.from_numpy(T).float().detach()

    def _generate_features(
        self,
        Ysets,  # An array of sets of Ylabels to scramble
        num_layers,
        num_intermediate,
        num_features,
        noise_std,
    ):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Generate random matrix common to all Ysets (train + test)
        transform_nn = dense_nn((self.num_states * self.num_actions * self.num_states), num_features, num_layers, num_intermediate, output_activation=None)

        # Generate training data by scrambling the Ys based on this matrix
        Xsets = []
        for Ys in Ysets:
            # Normalise data across the last dimension
            Ys_mean = Ys.reshape((-1, self.num_states, self.num_actions, self.num_states)).mean(dim=0)
            Ys_std = Ys.reshape((-1, self.num_states, self.num_actions, self.num_states)).std(dim=0)
            Ys_standardised = (Ys - Ys_mean) / (Ys_std + 1e-10)
            Ys_standardised = Ys_standardised.reshape((Ys.shape[0], Ys.shape[1],-1))
            assert not torch.isnan(Ys_standardised).any()

            # Encode Ys as features by multiplying them with a random matrix
            Xs = transform_nn(Ys_standardised).detach().clone()
            Xs_with_noise = Xs + torch.randn_like(Xs) * noise_std
            Xsets.append(Xs_with_noise)

        return (*Xsets,)

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs], [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs], [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return self.num_features, (self.num_states, self.num_actions, self.num_states)
    
    def get_output_activation(self):
        return 'softmax'

    def get_twostageloss(self):
        return 'mse'

    def _get_objective_exact(self, T, Pi, **kwargs):
        """
        Calculates the exact value of the policy by solving a set of linear equations.
        This complexity of this function is on the order of (num_states^num_arms states)^3.

        Not recommended for a large number of arms (>10)
        """
        # Create a vector of the joint state space across all arms (S_joint)
        S_joint = torch.tensor([state for state in product(range(self.num_states), repeat=self.num_arms)])

        # Create a vector of the joint action space across all arms (A_joint)
        A_joint = torch.tensor([state for state in product(range(self.num_actions), repeat=self.num_arms)])

        # Create a matrix of the joint rewards (R_joint) across all arms
        R_joint = S_joint.sum(dim=-1).float()

        # Create a matrix of the joint policy (Pi_joint) across all arms
        Pi_per_arm = Pi(S_joint)
        if T.ndim > 4:
            Pi_joint = torch.zeros((*Pi_per_arm.shape[:-2], S_joint.shape[0], A_joint.shape[0]))
        elif T.ndim == 4:
            Pi_joint = torch.zeros((S_joint.shape[0], A_joint.shape[0]))
        else:
            raise AssertionError()
        for idx, state_cur in enumerate(S_joint):
            for idy, act in enumerate(A_joint):
                    Pi_joint[..., idx, idy] = torch.stack([torch.abs((1 - act[n] - Pi_per_arm[..., idx, n])) for n in range(self.num_arms)]).prod(0)

        # Create a matrix of the joint tranisition probability (T_joint) across all arms
        # TODO: Speedup. This is the main bottleneck for DFL. (Vectorisation doesn't seem to work...)
        if T.ndim > 4:
            T_joint = torch.zeros((*T.shape[:-4], S_joint.shape[0], A_joint.shape[0], S_joint.shape[0]))
        elif T.ndim == 4:
            T_joint = torch.zeros((S_joint.shape[0], A_joint.shape[0], S_joint.shape[0]))
        else:
            raise AssertionError()
        for idx, state_cur in enumerate(S_joint):
            for idy, act in enumerate(A_joint):
                for idz, state_next in enumerate(S_joint):
                    T_joint[..., idx, idy, idz] = torch.stack([(T[..., n, state_cur[n], act[n], state_next[n]]) for n in range(self.num_arms)]).prod(0)

        # Solve for the value function
        A = (torch.eye(S_joint.shape[0]) - self.gamma * (Pi_joint.unsqueeze(-1) * T_joint).sum(-2))
        b = R_joint
        Vs = torch.linalg.solve(A, b)

        # Return the value function associated with [1, 1, ..., 1]
        #   (Because we start with everyone adhering)
        return Vs[..., -1]
    
    def _get_objective_sampled(self,
        T,
        Pi,
        num_timesteps=50,
        num_traj=25,
        isTrain=False,
    ):
        # Generate trajectory
        #   Initialisation
        rewards = []
        actions = []
        states  = []
        all_probs = []
        #   Generate start state
        #   We assume that everyone starts by adhering
        state = torch.ones(num_traj, *T.shape[:-4], self.num_arms).long()
        T_traj = T.unsqueeze(0).expand(num_traj, *T.shape)
        for _ in range(num_timesteps):
            # calculate probabilities of taking each action
            probs = Pi(state)
            all_probs.append(probs)

            # sample an action from that set of probs
            action_sampler = Categorical(probs)
            action = action_sampler.sample()
            action_onehot = one_hot(action, num_classes=self.num_arms)

            # use that action to update the state
            T_traj_state = gather_incomplete_left(T_traj, state)
            T_traj_state_action = gather_incomplete_left(T_traj_state, action_onehot)
            next_state_sampler = Categorical(T_traj_state_action)
            next_state = next_state_sampler.sample() 

            # get reward
            reward = state.sum(-1)

            # store state, action and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # preprocess states and actions
        states = torch.stack(states, dim=-2).float()
        actions = torch.stack(actions, dim=-1)

        # preprocess rewards
        rewards = torch.stack(rewards, dim=-1).float()
        #   calculate rewards to go for less variance
        for i in range(num_timesteps):
            discounting = self.gamma**torch.arange(i, num_timesteps)
            rewards[..., i] = (rewards[..., i:] * discounting).sum(-1)

        if not isTrain:
            # Return the average "return"
            output = rewards
        else:
            # Return the "pseudo-loss" = 
            probs = torch.stack(all_probs, dim=-2)
            sampler = Categorical(probs)
            log_probs = sampler.log_prob(actions)
            output = log_probs * rewards   # "pseudo-loss" that when differentiated with autograd gives the gradient of J(θ)

        output_reshaped = output.permute((*np.roll(list(range(len(T.shape[:-4]) + 1)), -1), len(T.shape[:-4]) + 1))
        return output_reshaped.mean(-1).mean(-1)

    def get_objective(self, T, Pi, **kwargs):
        """
        For a given policy (Pi), returns the expected return for that policy.
        """
        # Sanity check inputs
        assert T.shape[-4:] == (self.num_arms, self.num_states, self.num_actions, self.num_states)

        if self.eval_method == 'exact':
            obj = self._get_objective_exact
        elif self.eval_method == 'sim':
            obj = self._get_objective_sampled
        elif self.eval_method == 'ope':
            obj = self._get_objective_ope
        else:
            raise AssertionError('Invalid eval_method')

        return obj(T, Pi, **kwargs)

    def get_decision(
        self,
        Y,
        isTrain=False,
        **kwargs
    ):
        return self.opt_train(Y, self.gamma) if isTrain else self.opt_test(Y, self.gamma)


# Unit test for RMAB domain
if __name__=="__main__":
    # Initialise
    problem = RMAB()
    _, Y_train, _ = problem.get_train_data()

    # Generate a fake prediction
    T_rand = torch.rand_like(Y_train)
    T_rand /= T_rand.sum(-1, keepdim=True)
    # Solve for a policy with this fake prediction
    Pi_rand = problem.get_decision(T_rand, isTrain=True)
    # Evaluate the quality of the obtained policy
    J_rand = problem.get_objective(Y_train, Pi_rand)
    print(f"Return for Random Prediction: {J_rand.mean().item()}")

    # Solve for a policy with the true prediction
    Pi_true = problem.get_decision(Y_train, isTrain=False)
    # Evaluate the quality of the best possible prediction
    J_true = problem.get_objective(Y_train, Pi_true)
    print(f"Return for Optimal Prediction: {J_true.mean().item()}")
