import torch
import copy
import torch.nn.functional as F
import copy

from models.DDPG import Actor
from models.DDPG import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            lr=1e-3
    ):


        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state).clamp(-self.max_action, self.max_action)
            target_Q = reward + self.discount * not_done * self.critic_target(next_state, next_action)

        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        state = {
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict()
        }
        return state

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_target = copy.deepcopy(self.actor)