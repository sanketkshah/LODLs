from pdb import main, set_trace
from PThenO import PThenO
import pickle
import networkx as nx
import random
import numpy as np
from SubmodularOptimizer import SubmodularOptimizer
import torch
import metis
import pdb
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class BipartiteMatching(PThenO):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=110,  # number of instances to use from the dataset to train
        num_test_instances=25,  # number of instances to use from the dataset to test
        num_nodes=10,  # number of nodes in the LHS and RHS of the bipartite matching graphs
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(BipartiteMatching, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        # Load train and test labels
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.num_nodes = num_nodes
        self.Xs, self.Ys = self._load_instances(self.num_train_instances, self.num_test_instances, self.num_nodes)
        # self.Xs = torch.load('data/cora_features_bipartite.pt').reshape((27, 50, 50, 2866))
        # self.Ys = torch.load('data/cora_graphs_bipartite.pt').reshape((27, 50, 50))

        # Split data into train/val/test
        assert 0 < val_frac < 1
        self.val_frac = val_frac

        idxs = list(range(self.num_train_instances + self.num_test_instances))
        random.shuffle(idxs)
        self.val_idxs = idxs[0:int(self.val_frac * self.num_train_instances)]
        self.train_idxs = idxs[int(self.val_frac * self.num_train_instances):self.num_train_instances]
        self.test_idxs = idxs[self.num_train_instances:]
        assert all(x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs])

        # Create functions for optimisation
        self.opt_train = self._create_cvxpy_problem(isTrain=True)
        self.opt_test = self._create_cvxpy_problem(isTrain=False)

        # Undo random seed setting
        self._set_seed()

    def _load_instances(
        self,
        num_train_instances,
        num_test_instances,
        num_nodes,
        random_split=True,
        verbose=False,
    ):
        """
        Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        """
        # Load the labels dataset
        g = nx.read_edgelist('data/cora.cites')
        g = g.to_directed()  # remove directionality to make the problem easier
        nodes_before = [int(v) for v in g.nodes()]
        g = nx.convert_node_labels_to_integers(g, first_label=0)

        # Whittle the dataset down to the right size
        #   Initialise constants
        total_nodes = len(nodes_before)
        num_subsets = num_train_instances + num_test_instances
        assert num_subsets <= total_nodes // (num_nodes * 2)

        #   Whittle (coarse-grained)
        _, mapping = metis.part_graph(g, nparts=total_nodes // (num_nodes * 2))
        g_part = [nx.Graph(nx.subgraph(g, list(np.where(np.array(mapping) == i)[0]))) for i in range(num_subsets)]

        #   Ensure each part has n_nodes * 2 nodes
        nodes_available = []
        for i in range(num_subsets):
            if len(g_part[i]) > num_nodes * 2:
                degrees = [g_part[i].degree(v) for v in g_part[i]]
                order = np.argsort(degrees)
                nodes = np.array(g_part[i].nodes())
                num_remove = len(g_part[i]) - num_nodes * 2
                to_remove = nodes[order[:num_remove]]
                g_part[i].remove_nodes_from(to_remove)
                nodes_available.extend(to_remove)

        for i in range(num_subsets):
            if len(g_part[i]) < num_nodes * 2:
                num_needed =  num_nodes * 2 - len(g_part[i])
                to_add = nodes_available[:num_needed]
                nodes_available = nodes_available[num_needed:]
                g_part[i].add_nodes_from(to_add)

        # Load the features dataset
        features = np.loadtxt('data/cora.content')
        features_idx = features[:, 0]
        features = features[:, 1:]

        # Put it all together and format it correctly
        Xs = []
        Ys = []
        percent_removed = []
        for i in range(num_subsets):
            assert len(g_part[i].nodes()) == num_nodes * 2

            # Split nodes into LHS and RHS
            #   Create a split
            if random_split==True:
                part_nodes = list(g_part[i].nodes())
                random.shuffle(part_nodes)
                lhs_nodes, rhs_nodes = part_nodes[:num_nodes], part_nodes[num_nodes:]
                lhs_nodes_idx, rhs_nodes_idx = [list(g_part[i].nodes()).index(n) for n in lhs_nodes], [list(g_part[i].nodes()).index(n) for n in rhs_nodes]
            else:
                part_nodes = np.array(g_part[i].nodes())
                _, split = metis.part_graph(nx.complement(g_part[i]))
                if (np.array(split)==0).sum() != 50:  # if
                    num_flips = abs(50 - (np.array(split)==0).sum())

                lhs_nodes_idx, rhs_nodes_idx = np.where(np.array(split) == 0)[0], np.where(np.array(split) == 1)[0]
                lhs_nodes, rhs_nodes = part_nodes[lhs_nodes_idx], part_nodes[rhs_nodes_idx]

            #   Split the graph into 2 parts
            adj = nx.to_numpy_array(g_part[i])
            sum_before = adj.sum()
            adj = adj[lhs_nodes_idx]
            adj = adj[:, rhs_nodes_idx]
            Ys.append(adj)

            #   Diagnostic/Sanity Check
            edges_before = sum_before/2 + 1e-5
            percent_removed.append((edges_before - adj.sum())/edges_before)
            if verbose:
                print(sum_before/2, adj.sum())

            #   Get features
            feature_idxs_lhs = [int(np.where(features_idx == nodes_before[v])[0][0]) for v in lhs_nodes]
            feature_idxs_rhs = [int(np.where(features_idx == nodes_before[v])[0][0]) for v in rhs_nodes]
            feature_array = [[np.concatenate([features[idx], features[idx_other]]) for idx_other in feature_idxs_rhs] for idx in feature_idxs_lhs]
            Xs.append(feature_array)
            
        return torch.Tensor(np.array(Xs)), torch.Tensor(np.array(Ys))

    def get_train_data(self):
        return self.Xs[self.train_idxs], self.Ys[self.train_idxs], [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs[self.val_idxs], self.Ys[self.val_idxs], [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs[self.test_idxs], self.Ys[self.test_idxs], [None for _ in range(len(self.test_idxs))]
    
    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1
    
    def get_output_activation(self):
        return 'sigmoid'

    def get_twostageloss(self):
        return 'ce'

    def get_objective(self, Y, Z, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        assert Y.shape[-2:] == Z.shape[-2:]

        return torch.sum(Y * Z, dim=(-2, -1))

    def get_decision(
        self,
        Y,
        isTrain=False,
        max_instances_per_batch=5000,
        **kwargs
    ):
        # Split Y into reasonably sized chunks so that we don't run into memory issues
        # Assumption Y is only 3D at max
        assert Y.ndim in [2,3]
        if Y.ndim == 3:
            results = []
            for start in range(0, Y.shape[0], max_instances_per_batch):
                end = min(Y.shape[0], start + max_instances_per_batch)
                result = self.opt_train(Y[start:end])[0] if isTrain else self.opt_test(Y[start:end])[0]
                results.append(result)
            return torch.cat(results, dim=0)
        else:
            return self.opt_train(Y)[0] if isTrain else self.opt_test(Y)[0]

    def _create_cvxpy_problem(
        self,
        isTrain=True,
        gamma=0.1,
    ):
        # Variables
        Z = cp.Variable((self.num_nodes, self.num_nodes), nonneg=True)
        Y = cp.Parameter((self.num_nodes, self.num_nodes))

        # Objective
        matching_obj = cp.sum(cp.multiply(Z, Y))
        reg = cp.norm(Z) if isTrain else 0
        objective = cp.Maximize(matching_obj - gamma * reg)

        # Flow Constraints
        constraints = [cp.sum(Z, axis=0) == 1, cp.sum(Z, axis=1) == 1]

        # Problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        return CvxpyLayer(problem, parameters=[Y], variables=[Z])   

    # def _create_constraint_matrix(self):
    #     """
    #     Creates a matrix representation for the inequality constraints
    #     Gx <= h specific to the bipartite matching problem.

    #     Variables:
    #     #LHS nodes = #RHS nodes = self.num_nodes
    #     n_vars: (num_nodes * num_nodes) vector
    #     n_constraints = n_vars + #LHS nodes + #RHS nodes

    #     Output:
    #     G: (n_constraints, n_vars) matrix
    #     h: (n_constraints,) vector
    #     """
    #     # Create flow constraints
    #     #   Create constraints for RHS nodes
    #     #   \sum_i z_ij <= 1, forall j 
    #     G_lhs = torch.eye(self.num_nodes).repeat(1, self.num_nodes)
    #     h_lhs = torch.ones(self.num_nodes)

    #     #   Create constraints for LHS nodes
    #     #   \sum_j z_ij <= 1, forall i 
    #     G_rhs = G_lhs.detach().clone()
    #     G_rhs = G_rhs.reshape((self.num_nodes, self.num_nodes, self.num_nodes)).permute((0, 2, 1)).reshape((self.num_nodes, -1))
    #     h_rhs = torch.ones(self.num_nodes)

    #     # Create non-negativity constraints
    #     G_ineq = -torch.eye(self.num_nodes * self.num_nodes)
    #     h_ineq = torch.zeros(self.num_nodes * self.num_nodes)

    #     # Putting them together
    #     G = torch.cat((G_lhs, G_rhs, G_ineq))
    #     h = torch.cat((h_lhs, h_rhs, h_ineq))

    #     return G, h

if __name__=="__main__":
    pdb.set_trace()
    problem = BipartiteMatching()