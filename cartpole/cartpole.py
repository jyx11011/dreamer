#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

class Cartpole(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 5
        self.n_ctrl = 1

        self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 1)))
        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.x_threshold = 0.25
        self.max_velocity = 10

        self.dt = 0.01 
        self.force_mag=0.25
        self.lower = -self.force_mag
        self.upper = self.force_mag

        # 0  1      2        3   4
        # x dx cos(th) sin(th) dth
        self.goal_state = torch.Tensor(  [ 0.,  0.,  -1., 0.,   0.])
        self.goal_weights = torch.Tensor([0.1, 0.1,  1., 1., 0.1])
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        u = torch.clamp(u[:,], -self.force_mag, self.force_mag)

        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        th = torch.atan2(sin_th, cos_th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        th_acc = (gravity * sin_th + cos_th * cart_in) / \
                 (length * (4./3. - masspole * cos_th**2 /
                                     total_mass))
        xacc = cart_in + polemass_length * th_acc * cos_th / total_mass

        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        state = torch.stack((
            x, dx, torch.cos(th), torch.sin(th), dth
        ),1)

        return state

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)

