import gym
import numpy as np
import torch
import torch.autograd
from mpc import mpc

class Dynamics(torch.nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self._dynamics=dynamics

    def forward(self, state, action):
        return self._dynamics.transition(state, action)

class MPC_planer:
    def __init__(self, timesteps, n_batch, nx, nu, dynamics,
            goal_weights=None, ctrl_penalty=0.001, iter=5,
            action_low=None, action_high=None, eps=0.01):
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = torch.ones([timestamps, n_batch, nu]) * action_low
        self._action_high = torch.ones([timestamps, n_batch, nu]) * action_high
        self._eps = eps
        dtype=torch.float64

        if goal_weights is None:
            goal_weights = torch.ones(nx, dtype=dtype)
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu, dtype=dtype)
        ))
        self._Q = torch.diag(q).repeat(timesteps, n_batch, 1, 1)  # T x B x nx+nu x nx+nu
        self._dynamics = Dynamics(dynamics)

    def set_goal_state(self, goal_state=None):
        px = -torch.sqrt(self.goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(nu, dtype=dtype)))
        p = p.repeat(timesteps, n_batch, 1)
        self._cost = mpc.QuadCost(self._Q, p)
        self._u_init = None

    def get_next_action(self, state):
        state = state.copy()
        state = torch.tensor(state, dtype=dtype).view(1, -1)
        
        ctrl = mpc.MPC(self.nx, self.nu, timesteps, u_lower=self._action_low, u_upper=self._action_high, lqr_iter=self._iter,
                        exit_unconverged=False, eps=self._eps,
                        n_batch=n_batch, backprop=False, verbose=0, u_init=self._u_init,
                        grad_method=mpc.GradMethods.AUTO_DIFF)

        nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[0] 
        self._u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, nu, dtype=dtype)), dim=0)

        return action