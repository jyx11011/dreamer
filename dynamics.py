import gym
import numpy as np
import tensorflow as tf
import torch
import torch.autograd
from mpc import mpc

class Dynamics(torch.nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self._dynamics=dynamics

    def forward(self, state, action):
        tf_state = tf.convert_to_tensor([state.numpy()])
        tf_action = tf.convert_to_tensor([action.numpy()])
        next_state = self._dynamics.transition(state, action)
        return torch.tensor(next_state.numpy(), dtype=torch.float16)

class MPC_planer:
    def __init__(self, timesteps, n_batch, nx, nu, dynamics,
            goal_weights=None, ctrl_penalty=0.001, iter=5,
            action_low=None, action_high=None, eps=0.01):
        self._timesteps = timesteps
        self._n_batch = n_batch
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = torch.ones([timesteps, n_batch, nu]) * action_low
        self._action_high = torch.ones([timesteps, n_batch, nu]) * action_high
        self._eps = eps
        self._dtype=torch.float16

        if goal_weights is None:
            goal_weights = torch.ones(nx, dtype=torch.float64)
        self._goal_weights = goal_weights
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu, dtype=torch.float64)
        ))
        self._Q = torch.diag(q).repeat(timesteps, n_batch, 1, 1).type(torch.float16)  # T x B x nx+nu x nx+nu
        self._dynamics = Dynamics(dynamics)

    def set_goal_state(self, state):
        state = state.numpy()
        goal_state = torch.tensor(state, dtype=self._dtype).view(1, -1)[0]
        px = -torch.sqrt(self._goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(self._nu, dtype=self._dtype)))
        p = p.repeat(self._timesteps, self._n_batch, 1)
        self._cost = mpc.QuadCost(self._Q, p)
        self._u_init = None

    def get_next_action(self, state):
        state = state.numpy()
        state = torch.tensor(state, dtype=self._dtype).view(1, -1)
        
        ctrl = mpc.MPC(self._nx, self._nu, self._timesteps, 
                        u_lower=self._action_low, u_upper=self._action_high, 
                        lqr_iter=self._iter, eps=self._eps, n_batch=1,
                        u_init=self._u_init,
                        exit_unconverged=False, backprop=False, verbose=0, 
                        grad_method=mpc.GradMethods.FINITE_DIFF)

        nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[0] 
        self._u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self._n_batch, self._nu, dtype=self._dtype)), dim=0)

        return action
