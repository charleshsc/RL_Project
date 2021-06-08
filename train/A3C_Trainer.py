import torch.multiprocessing as mp

from models.A3C import ActorCritic
from workers.A3C_worker import A3C_Worker
from optim.SharedAdam import SharedAdam


def A3C_Trainer(logger, saver, cfg_dict, checkpoint):
    global_net = ActorCritic(cfg_dict.get('state_dimention'), cfg_dict.get('action_dimention'), cfg_dict.get('max_action_value'))
    global_net.share_memory() # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=cfg_dict.get('lr'), betas=(0.95, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i',0), mp.Value('d',0) , mp.Queue()

    workers = [A3C_Worker(global_net,optimizer,global_ep,global_ep_r,res_queue,i, cfg_dict, logger, saver) for i in range(mp.cpu_count())]
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
    },filename='final_model')
    saver.save_evaluation(res)