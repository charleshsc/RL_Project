import os
import shutil

import numpy as np
import torch
from collections import OrderedDict
import glob
import torch.distributed as dist

class Saver(object):

    def __init__(self, save_direcotry, args):
        self.experiment_dir = save_direcotry
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.args = args
        self.save_experiment_config(args)
        self.best_reward = 0

    def save_checkpoint(self, state, filename='checkpoint.pth', reward=0):
        """Saves checkpoint to disk"""
        if filename.split('.')[-1] != 'pth':
            filename = filename + '.pth'
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if reward > self.best_reward:
            self.best_reward = reward
            filename = os.path.join(self.experiment_dir, "best_model.pth")
            print("save the best model with reward %f" % reward)
            torch.save(state, filename)


    def save_experiment_config(self, args):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        for key in vars(args):
            log_file.write(key + ':' + str(getattr(args, key)) + '\n')
        log_file.close()

    def save_evaluation(self, evaluation, filename='evaluation.npy'):
        np.save(os.path.join(self.experiment_dir, filename), evaluation)
