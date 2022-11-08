import random
import pdb
import torch
import torch.nn.functional as F
import numpy as np

from utils import gather_incomplete_left, trim_left, solve_lineqn


class RMABSolver(torch.nn.Module):
    """
    Solves for the Whittle Index policy for RMABs with 2-states and 2-actions.
    """
    def __init__(
        self,
        budget,  # the number of arms that can be selected per round
        isTrain=True, # variable indicating whether this is training or test
    ):
        super(RMABSolver, self).__init__()
        self.budget = budget
        self.isTrain = isTrain
        if isTrain:
            self.soft_topk = TopK_custom(budget)

    def _get_whittle_indices(self, T, gamma):
        '''
        Source: https://github.com/armman-projects/Google-AI/
        Inputs:
            Ts: Transition matrix of dimensions ... X 2 X 2 X 2 where axes are:
                ... (batch), start_state, action, end_state
            gamma: Discount factor
        Returns:
            index: ... X 2 Tensor of Whittle index for states (0,1)
        '''
        # Matrix equations for state 0
        row1_s0  =   torch.stack([torch.ones_like(T[...,0,0,0]) , gamma * T[...,0,0,0] - 1, gamma * T[...,0,0,1]    ], -1)
        row2_s0  =   torch.stack([torch.zeros_like(T[...,0,1,0]), gamma * T[...,0,1,0] - 1, gamma * T[...,0,1,1]    ], -1)
        row3a_s0 =   torch.stack([torch.ones_like(T[...,1,0,0]) , gamma * T[...,1,0,0]    , gamma * T[...,1,0,1] - 1], -1)
        row3b_s0 =   torch.stack([torch.zeros_like(T[...,1,1,0]), gamma * T[...,1,1,0]    , gamma * T[...,1,1,1] - 1], -1)

        A1_s0 = torch.stack([row1_s0, row2_s0, row3a_s0], -2)
        A2_s0 = torch.stack([row1_s0, row2_s0, row3b_s0], -2)
        b_s0 = torch.tensor([0,0,-1], dtype=torch.float32)

        # Matrix equations for state 1
        row1_s1  =   torch.stack([torch.ones_like(T[...,1,0,0]) , gamma * T[...,1,0,0]    , gamma * T[...,1,0,1] - 1], -1)
        row2_s1  =   torch.stack([torch.zeros_like(T[...,1,1,0]), gamma * T[...,1,1,0]    , gamma * T[...,1,1,1] - 1], -1)
        row3a_s1 =   torch.stack([torch.ones_like(T[...,0,0,0]) , gamma * T[...,0,0,0] - 1, gamma * T[...,0,0,1]    ], -1)
        row3b_s1 =   torch.stack([torch.zeros_like(T[...,0,1,0]), gamma * T[...,0,1,0] - 1, gamma * T[...,0,1,1]    ], -1)

        A1_s1 = torch.stack([row1_s1, row2_s1, row3a_s1], -2)
        A2_s1 = torch.stack([row1_s1, row2_s1, row3b_s1], -2)
        b_s1 = torch.tensor([-1,-1,0], dtype=torch.float32)

        # Compute candidate whittle indices
        cnd1_s0 = solve_lineqn(A1_s0, b_s0)
        cnd2_s0 = solve_lineqn(A2_s0, b_s0)

        cnd1_s1 = solve_lineqn(A1_s1, b_s1)
        cnd2_s1 = solve_lineqn(A2_s1, b_s1)

        # TODO: Check implementation. Getting WI > 1??
        ## Following line implements condition checking when candidate1 is correct
        ## It results in an array of size N, with value 1 if candidate1 is correct else 0.
        cand1_s0_mask = (cnd1_s0[..., 0] + 1.0) + gamma * (T[...,1,0,0] * cnd1_s0[...,1] + T[...,1,0,1] * cnd1_s0[...,2]) >= \
                            1.0 + gamma * (T[...,1,1,0] * cnd1_s0[...,1] + T[...,1,1,1] * cnd1_s0[...,2])
        cand1_s1_mask = (cnd1_s1[..., 0])       + gamma * (T[...,0,0,0] * cnd1_s1[...,1] + T[...,0,0,1] * cnd1_s1[...,2]) >= \
                                  gamma * (T[...,0,1,0] * cnd1_s1[...,1] + T[...,0,1,1] * cnd1_s1[...,2])

        cand2_s0_mask = ~cand1_s0_mask
        cand2_s1_mask = ~cand1_s1_mask

        return torch.stack([cnd1_s0[..., 0] * cand1_s0_mask + cnd2_s0[..., 0] * cand2_s0_mask, cnd1_s1[..., 0] * cand1_s1_mask + cnd2_s1[..., 0] * cand2_s1_mask], -1)


    def forward(
        self,
        T,  # predicted transition probabilities
        gamma,  # discount factor
    ):
        # Make sure the shape is correct
        assert T.shape[-3:] == (2, 2, 2)
        if T.ndim == 4:
            T = T.unsqueeze(0)
        assert T.ndim == 5

        # Get whittle indices
        W = self._get_whittle_indices(T, gamma)

        # Define policy function
        def pi(
            state,  # a vector denoting the current state
        ):
            # Preprocessing
            state = trim_left(state)
            W_temp = trim_left(W)

            # Find the number of common dimensions between W and state
            common_dims = 0
            while common_dims < min(W_temp.ndim - 1, state.ndim) and W_temp.shape[-(common_dims + 2)] == state.shape[-(common_dims + 1)]:
                common_dims += 1
            assert common_dims > 0  # ensures that num_arms is consistent across W and state
            assert state.max() < W_temp.shape[-1] and state.min() >= 0

            # Enable broadcasting
            #   Expand state
            if W_temp.ndim > common_dims + 1 and state.ndim == common_dims:
                for i in range(common_dims + 2, W_temp.ndim + 1):
                   state = state.unsqueeze(0).expand(W_temp.shape[-i], *state.shape)
            #   Expand W
            elif state.ndim > common_dims and W_temp.ndim == common_dims + 1:
                for i in range(common_dims + 1, state.ndim + 1):
                    W_temp = W_temp.unsqueeze(0).expand(state.shape[-i], *W_temp.shape)
            #   Expand both
            elif state.ndim > common_dims and W_temp.ndim > common_dims + 1:
                # Special case for get_obj_exact: We want to calculate the policy for all states and Ws
                if  W_temp.ndim == 3 and state.ndim == 2 and W_temp.shape[-3] != state.shape[-2]:
                    state = state.unsqueeze(0).expand(W_temp.shape[-3], *state.shape)
                    W_temp = W_temp.unsqueeze(1).expand(W_temp.shape[0], state.shape[1], *W_temp.shape[1:])
                else:
                    raise AssertionError("Invalid shapes")            
            
            # Get whittle indices for the relevant states
            W_state = gather_incomplete_left(W_temp, state)

            # Choose action based on policy
            if self.isTrain:
                gamma = self.soft_topk(-W_state)
                act = gamma[...,0] * W_state.shape[-1]
            else:
                _, idxs = torch.topk(W_state, self.budget)
                act = torch.nn.functional.one_hot(idxs.squeeze(), W_state.shape[-1])
            return act
        return pi


class TopK_custom(torch.nn.Module):
    """
    Source: Code from https://proceedings.neurips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf 
    """
    def __init__(self, k, epsilon=0.01, max_iter = 200):
        super(TopK_custom, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1]).to(device)
        self.max_iter = max_iter

    def forward(self, scores):
        n = scores.shape[-1]
        bs = scores.shape[:-1]
        scores = scores.view([*bs, n, 1])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        anchors = self.anchors.view(*((1,)*(len(bs) + 1)), 2)
        C_raw = (scores-anchors)**2
        C = C_raw / C_raw.amax(dim=(-2, -1), keepdim=True)
        
        mu = torch.ones([*bs, n, 1], requires_grad=False).to(device)/n
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nu = torch.FloatTensor([self.k/n, (n-self.k)/n]).view((*((1,)*(len(bs) + 1)), 2)).to(device)
        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)
        
        return Gamma


class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):
        bs, n, k_ = C.shape[:-2], C.shape[-2], C.shape[-1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        f = torch.zeros([*bs, n, 1]).to(device)
        g = torch.zeros([*bs, 1, k_]).to(device)

        epsilon_log_mu = epsilon*torch.log(mu)
        epsilon_log_nu = epsilon*torch.log(nu)
        
        def min_epsilon_row(Z, epsilon):
            return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)
        
        def min_epsilon_col(Z, epsilon):
            return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)

        for _ in range(max_iter):
            f = min_epsilon_row(C-g, epsilon)+epsilon_log_mu
            g = min_epsilon_col(C-f, epsilon)+epsilon_log_nu

        Gamma = torch.exp((-C+f+g)/epsilon)
        ctx.save_for_backward(mu, nu, Gamma)
        ctx.epsilon = epsilon      
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        #Gamma [*bs, n, k+1]
                  
        with torch.no_grad():
            nu_ = nu[...,:-1]
            Gamma_ = Gamma[...,:-1]
    
            bs, n, k_ = Gamma.shape[:-2], Gamma.shape[-2], Gamma.shape[-1]
            
            inv_mu = 1./(mu.view([1,-1]))  #[1, n]
            Kappa = torch.diag_embed(nu_.squeeze(-2)) \
                    -torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)   #[*bs, k, k]
            #print(Kappa, Gamma_)
            padding_value = 1e-10
            ridge = torch.ones([*bs, k_-1]).diag_embed()
            inv_Kappa = torch.inverse(Kappa+ridge*padding_value) #[*bs, k, k]
            #print(Kappa, inv_Kappa) 
            mu_Gamma_Kappa = (inv_mu.unsqueeze(-1)*Gamma_).matmul(inv_Kappa) #[*bs, n, k]
            H1 = inv_mu.diag_embed() + mu_Gamma_Kappa.matmul(Gamma_.transpose(-1, -2))*inv_mu.unsqueeze(-2) #[*bs, n, n]
            H2 = - mu_Gamma_Kappa  #[*bs, n, k]
            H3 = H2.transpose(-1,-2) #[*bs, k, n]
            H4 = inv_Kappa #[*bs, k, k]
    
            H2_pad = F.pad(H2, pad=(0, 1), mode='constant', value=0) 
            H4_pad = F.pad(H4, pad=(0, 1), mode='constant', value=0)
            grad_f_C =  H1.unsqueeze(-1)*Gamma.unsqueeze(-3) \
                       + H2_pad.unsqueeze(-2)*Gamma.unsqueeze(-3) #[*bs, n, n, k+1]
            grad_g_C =  H3.unsqueeze(-1)*Gamma.unsqueeze(-3) \
                       + H4_pad.unsqueeze(-2)*Gamma.unsqueeze(-3) #[*bs, k, n, k+1]
    
            grad_g_C_pad = F.pad(grad_g_C, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)
            grad_C1 = grad_output_Gamma * Gamma
            grad_C2 = torch.sum(grad_C1.view([*bs, n, k_, 1, 1])*grad_f_C.unsqueeze(-3), dim=(1,2))
            grad_C3 = torch.sum(grad_C1.view([*bs, n, k_, 1, 1])*grad_g_C_pad.unsqueeze(-4), dim=(1,2))
    
            grad_C = (-grad_C1+grad_C2+grad_C3)/epsilon
                   
        return grad_C, None, None, None, None


# Unit test for submodular optimiser
if __name__ == '__main__':
    # Make it reproducible
    rand_seed = 1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    # Function to generate random transition probabilities          
    def generate_instances(R, num_instances, num_arms, num_states, min_lift):
        T = np.zeros((num_instances, num_arms, num_states, 2, num_states))
        for i in range(num_instances): 
            for j in range(num_arms):
                for k in range(num_states):
                    while True:
                        passive_transition = np.random.rand(num_states)
                        passive_transition /= passive_transition.sum()
                        active_transition  = np.random.rand(num_states)
                        active_transition /= active_transition.sum()
                        if active_transition @ R > passive_transition @ R + min_lift: # Ensure that calling is significantly better
                            T[i,j,k,0,:] = passive_transition
                            T[i,j,k,1,:] = active_transition
                            break
        return torch.from_numpy(T).float().detach()
    
    R = np.arange(2)
    T = generate_instances(R, 2, 5, 2, 0.2).requires_grad_()
    opt = RMABSolver(budget=1)

    # Perform backward pass
    pdb.set_trace()
    state = torch.bernoulli(0.5 * torch.ones(2, 5))
    pi = opt(T, 0.99)
    act = pi(state)

    # Check gradients
    loss = act.square().sum()
    optimizer = torch.optim.Adam([T], lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
