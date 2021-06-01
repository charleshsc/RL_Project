import torch
import gym
import torch.multiprocessing as mp
import yaml
import os
import time

from models.A3C import ActorCritic
from workers.worker import Worker
from optim.SharedAdam import SharedAdam
from utils.logger import get_logger
from utils.args import obtain_env_args
from utils.Saver import Saver


def Trainer(logger, saver, cfg_dict):
    global_net = ActorCritic(cfg_dict.get('state_dimention'), cfg_dict.get('action_dimention'))
    global_net.share_memory() # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=args.lr, betas=(0.95, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i',0), mp.Value('d',0) , mp.Queue()

    workers = [Worker(global_net,optimizer,global_ep,global_ep_r,res_queue,i, cfg_dict, logger) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    saver.save_checkpoint(state={
        'state_dict' : global_net.state_dict()
    })


if __name__ == '__main__':
    args = obtain_env_args()
    CFG_FILE = args.env_name + '.yaml'
    with open(CFG_FILE, 'r') as cfg_file:
        cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    logger = get_logger(__name__, os.path.join(args.save,time_str+'.log'))

    saver = Saver(args.save, args)


