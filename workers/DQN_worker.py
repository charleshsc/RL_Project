import torch
import copy
import torch.nn.functional as F
import copy
import numpy as np

from models.DQN import DQN_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            discount = 0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            target_update_step=100,
            algo = 'DQN'
    ):
        self.Q = DQN_model(state_dim, action_dim).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)

        self.action_dim = action_dim
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_step = target_update_step
        self.algo = algo

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            action = self.Q(state).argmax()
            return action.detach().cpu().numpy()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            if self.algo == 'DQN':
                q_target = self.Q_target(next_state)
            elif self.algo == 'DDQN':
                q2 = self.Q(next_state)
                q_target = self.Q_target(next_state)
                q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))

            q_backup = reward.squeeze(1) + self.discount * not_done.squeeze(1) * q_target.max(1)[0]

        q = self.Q(state).gather(1, action.long()).squeeze(1)

        q_loss = F.mse_loss(q, q_backup)
        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        if self.total_it % self.target_update_step == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self):
        state = {
            'Q_state_dict' : self.Q.state_dict(),
            'Q_optimizer_state_dict' : self.Q_optimizer.state_dict(),
        }
        return state

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.Q_optimizer.load_state_dict(checkpoint['Q_optimizer_state_dict'])
        self.Q_target = copy.deepcopy(self.Q)




