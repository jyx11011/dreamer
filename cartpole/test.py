from dm_control import suite

import logging
import math
import time
import os
import gym
import numpy as np
import torch
import torch.autograd
from cartpole import Cartpole
from gym import wrappers, logger as gym_log
from mpc import mpc

def get_state(observation):
    obs=dict(observation)
    pos=obs['position']
    vel=obs['velocity']
    return torch.Tensor([[pos[0], vel[0], pos[1], pos[2], vel[1]]])

if __name__ == "__main__":
    os.environ['MUJOCO_GL'] = 'egl'
    TIMESTEPS = 50  # T
    N_BATCH = 1
    LQR_ITER = 5

    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)

    env = suite.load(domain_name="cartpole", task_name="swingup")
    action_spec = env.action_spec()

    dynamics=Cartpole()
    nx = 5
    nu = 1

    u_init = None
    render = True
    run_iter = 500

    q = torch.cat((
        dynamics.goal_weights,
        dynamics.ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(dynamics.goal_weights) * dynamics.goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    time_step = env.reset()
    total_reward = 0
    for i in range(run_iter):
        state = get_state(time_step.observation)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=dynamics.lower, u_upper=dynamics.upper, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=dynamics.mpc_eps,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF,
                       linesearch_decay=dynamics.linesearch_decay)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, dynamics)
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        action = np.ones(action_spec.shape)*action[0,0].item()
        time_step = env.step(action)
        total_reward += time_step.reward
        if render:
            env.physics.render(64,64)
print(get_state(time_step.observation))
print(total_reward)

#export MUJOCO_GL=egl