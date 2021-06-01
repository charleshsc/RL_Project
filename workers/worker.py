import math
import random

import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from models.A3C import ActorCritic

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_epoch, global_epoch_r, res_queue, name, cfg_dict, logger):
        super(Worker, self).__init__()
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
                a = self.local_net.choose_action(torch.from_numpy(s[None,:])) ## None并不指代数组中的某一维，None用于改变数组的维度。
                s_, r, done, _ = self.env.step(a.clip(self.min_action_value,self.max_action_value))
                if t == self.cfg_dict.get('max_epoch_steps') - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1) # normalize  maybe differ according to env?

                if total_step % self.cfg_dict.get('update_global_iter') == 0 or done:
                    # sync

                    push_and_pull(self.opt, self.local_net, self.global_net, done, s_, buffer_s, buffer_a, buffer_r, self.cfg_dict.get('gamma'))
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.global_epoch, self.global_epoch_r, ep_r, self.res_queue, self.name, self.logger)
                        break

                s = s_
                total_step += 1

        self.res_queue.put(None)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(torch.from_numpy(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_(
        torch.from_numpy(np.vstack(bs)),
        torch.from_numpy(np.array(ba)),
        torch.from_numpy(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name, logger):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    logger.info(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )