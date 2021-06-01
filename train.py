import torch
import gym
import torch.multiprocessing as mp
import yaml
import os
import time
import numpy as np

from models.A3C import ActorCritic
from workers.A3C_worker import A3C_Worker
from optim.SharedAdam import SharedAdam
from utils.logger import get_logger
from utils.args import obtain_env_args
from utils.Saver import Saver
from utils.utils import ReplayBuffer
from eval.eval import eval_policy
from workers.TD3_worker import TD3


def A3C_Trainer(logger, saver, cfg_dict, checkpoint):
    global_net = ActorCritic(cfg_dict.get('state_dimention'), cfg_dict.get('action_dimention'))
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

def Trainer(logger, saver, cfg_dict, checkpoint):
    # init
    state_dimention = cfg_dict.get('state_dimention')
    action_dimention = cfg_dict.get('action_dimention')
    max_action_value = cfg_dict.get('max_action_value')
    seed = cfg_dict.get('seed')
    env_name = cfg_dict.get('env_name')
    start_timesteps = cfg_dict.get('start_timesteps')
    max_timesteps = cfg_dict.get('max_timesteps')
    expl_noise = cfg_dict.get('expl_noise')
    batch_size = cfg_dict.get('batch_size')
    discount = cfg_dict.get('discount')
    tau = cfg_dict.get('tau')
    policy_noise = cfg_dict.get('policy_noise') * max_action_value
    noise_clip = cfg_dict.get('noise_clip') * max_action_value
    policy_freq = cfg_dict.get('policy_freq')

    # define the model
    model = TD3(state_dimention, action_dimention, max_action_value,discount,tau,policy_noise, noise_clip, policy_freq)

    # Set seeds
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer(state_dimention, action_dimention, cfg_dict.get('capacity'))
    evaluations = [eval_policy(model, env_name, seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    model.select_action(np.array(state))
                    + np.random.normal(0, max_action_value * expl_noise, size=action_dimention)
            ).clip(-max_action_value, max_action_value)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            model.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            logger.info(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % cfg_dict.get('eval_freq') == 0:
            evaluations.append(eval_policy(model, env_name, seed))
            saver.save_evaluation(evaluations)
            saver.save_checkpoint(model.save())




if __name__ == '__main__':
    args = obtain_env_args()
    CFG_FILE = args.env_name + '.yaml'
    with open(CFG_FILE, 'r') as cfg_file:
        cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    logger = get_logger(__name__, os.path.join(args.save,time_str+'.log'))

    saver = Saver(args.save, args)
    checkpoint = args.checkpoint

    if args.algorithm == 'A3C':
        A3C_Trainer(logger, saver, cfg_dict.get('A3C'), checkpoint)
    elif args.algorithm == 'TD3':
        Trainer(logger, saver, cfg_dict.get('TD3'), checkpoint)


