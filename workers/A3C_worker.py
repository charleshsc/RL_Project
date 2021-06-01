import math
import random

import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from models.A3C import ActorCritic

class A3C_Worker(mp.Process):
    def __init__(self, gnet, opt, global_epoch, global_epoch_r, res_queue, name, cfg_dict, logger, saver):
        super(A3C_Worker, self).__init__()
        self.env_name = cfg_dict.get('env_name')
        self.state_dimention = cfg_dict.get('state_dimention')
        self.action_dimention = cfg_dict.get('action_dimention')
        self.name = 'Worker_%i' % name
        self.global_epoch, self.global_epoch_r, self.res_queue = global_epoch, global_epoch_r, res_queue
        self.global_net, self.opt = gnet, opt
        self.local_net = ActorCritic(self.state_dimention, self.action_dimention)
        self.env = gym.make(self.env_name).unwrapped
        self.cfg_dict = cfg_dict
        self.min_action_value = cfg_dict.get('min_action_value')
        self.max_action_value = cfg_dict.get('max_action_value')
        self.logger = logger
        self.saver = saver
        self.best_reward = 0
        assert self.state_dimention == self.env.observation_space.shape[0]
        assert self.action_dimention == self.env.action_space.shape[0]

    def run(self):
        total_step = 1
        while self.global_epoch.value < self.cfg_dict.get('max_epochs'):
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(self.cfg_dict.get('max_epoch_steps')):
                if self.name == 'Worker_0':
                    self.env.render()
                a = self.local_net.select_action(torch.tensor(s[None,:],dtype=torch.float))
                s_, r, done, _ = self.env.step(a.clip(self.min_action_value,self.max_action_value))
                if t == self.cfg_dict.get('max_epoch_steps') - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1) # normalize  maybe differ according to env?

                if total_step % self.cfg_dict.get('update_global_iter') == 0 or done:
                    # sync

                    self.push_and_pull(done, s_, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        self.record(ep_r)
                        break

                s = s_
                total_step += 1

        self.res_queue.put(None)

    def record(self, ep_r):
        with self.global_epoch.get_lock():
            self.global_epoch.value += 1
        with self.global_epoch_r.get_lock():
            if self.global_epoch_r.value == 0.:
                self.global_epoch_r.value = ep_r
            else:
                self.global_epoch_r.value = self.global_epoch_r.value * 0.99 + ep_r * 0.01
        self.res_queue.put(self.global_epoch_r.value)
        self.logger.info(
            self.name +
            " Ep:" + str(self.global_epoch.value) +
            "| Ep_r: %.0f" % self.global_epoch_r.value
        )

        if self.global_epoch.value % self.cfg_dict.get('epoch_to_save') == 0:
            self.saver.save_checkpoint(state={
                'state_dict': self.global_net.state_dict(),
                'reward' : ep_r
            },filename=self.name+"checkpoint")

    def push_and_pull(self,done, s_, bs, ba, br):
        if done:
            v_s_ = 0.  # terminal
        else:
            v_s_ = self.local_net.forward(torch.tensor(s_[None, :], dtype=torch.float))[-1].data.numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]:  # reverse buffer r
            v_s_ = r + self.cfg_dict.get('gamma') * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = self.local_net.loss_(
            torch.tensor(np.vstack(bs), dtype=torch.float),
            torch.tensor(np.vstack(ba), dtype=torch.float),
            torch.tensor(np.array(buffer_v_target)[:, None], dtype=torch.float))

        # calculate local gradients and push local parameters to global
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        # pull global parameters
        self.local_net.load_state_dict(self.global_net.state_dict())
