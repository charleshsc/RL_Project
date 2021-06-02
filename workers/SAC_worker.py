import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from models.SAC import Actor
from models.SAC import Q

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            lr=3e-4,
            discount=0.99,
            tau=0.005,
            log_std_max=2,
            log_std_min=-20,
            alpha=0.2,
            automatic_entropy_tuning=False,
            eval_mode=False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, log_std_max, log_std_min).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr= lr)

        self.q1 = Q(state_dim, action_dim).to(device)
        self.q2 = Q(state_dim, action_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self.q_optimizer = optim.Adam(self.q_parameters, lr=lr)

        self.distribution = torch.distributions.Normal
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.eval_mode = eval_mode
        self.alpha = alpha

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((action_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.eval_mode:
            action, _, _ = self.actor(state)
        else:
            _, action, _ = self.actor(state)
        action = action.detach().cpu().numpy()
        return action

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
        _, pi, log_pi = self.actor(state)
        _, next_pi, next_log_pi = self.actor(next_state)
        q1 = self.q1(state, action).squeeze(1)
        q2 = self.q2(state, action).squeeze(1)

        # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))), min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
        min_q_pi = torch.min(self.q1(state, pi), self.q2(state, pi)).squeeze(1)
        min_q_next_pi = torch.min(self.q1_target(next_state, next_pi), self.q2_target(next_state,next_pi)).squeeze(1)

        # Targets for Q regression
        v_backup = min_q_next_pi - self.alpha * next_log_pi
        q_backup = reward.squeeze(1) + self.discount * not_done.squeeze(1) * v_backup

        # SAC losses
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()
        q1_loss = F.mse_loss(q1, q_backup.detach())
        q2_loss = F.mse_loss(q2, q_backup.detach())
        q_loss = q1_loss + q2_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # soft update
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save(self):
        state = {
            'q1_state_dict' : self.q1.state_dict(),
            'q2_state_dict' : self.q2.state_dict(),
            'q_optimizer_state_dict' : self.q_optimizer.state_dict(),
            'actor_state_dict' : self.actor.state_dict(),
            'actor_optimizer_state_dict' : self.actor_optimizer.state_dict()
        }
        return state

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

