import math
import random

import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp


class ActorCritic(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, max_action_value=1):
        super(ActorCritic, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.actor_linear = nn.Linear(in_channels, 200)
        self.actor_mu = nn.Linear(200, out_channels) # estimated action value /均值

        self.sigma = nn.Linear(200, out_channels) # estimated variance /方差

        self.critic_lieanr = nn.Linear(in_channels, 100)
        self.v = nn.Linear(100, 1) # estimated value for state

        self._initialize_weights()

        self.distribution = torch.distributions.Normal # 正态分布
        self. max_action_value =  max_action_value

    def forward(self, x):
        actor = F.relu6(self.actor_linear(x))
        mu = 2 * torch.tanh(self.actor_mu(actor))
        sigma = F.softplus(self.sigma(actor)) + 0.001
        critic = F.relu6(self.critic_lieanr(x))
        values = self.v(critic)
        return mu, sigma, values

    def select_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        return m.sample().numpy().clip(-self. max_action_value,self. max_action_value)  #动作空间是连续的

    def loss_(self, s, a, v_t):
        # s: (bs,376), a: (bs,17), v_t: (bs,1)
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a).sum(dim=-1).unsqueeze(-1) # 取动作a的概率对数 如果是多维度的话，直接全部相加作为最后的概率 # (bs, 1)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale) # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()

        return total_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.)