import yaml
import torch

from eval.eval import eval_policy
from workers import *
from utils.args import obtain_env_args

if __name__ == '__main__':
    args = obtain_env_args()

    CFG_FILE = args.env_name + '.yaml'
    with open(CFG_FILE, 'r') as cfg_file:
        cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

    env_name = cfg_dict.get('env_name')
    state_dim = cfg_dict.get('state_dimention')
    action_dim = cfg_dict.get('action_dimention')
    max_action = cfg_dict.get('max_action_value')
    seed = cfg_dict.get('seed')

    checkpoint = args.checkpoint

    if args.algorithm == 'A3C':
        net = ActorCritic(state_dim, action_dim)
        checkpoint = torch.load(checkpoint)
        net.load_state_dict(checkpoint)
        eval_policy(net, env_name, seed, is_render=True)
    elif args.algorithm == 'TD3':
        net = TD3(state_dim,action_dim,max_action)
        net.load(checkpoint)
        eval_policy(net, env_name, seed, is_render=True)
    elif args.algorithm == 'SAC':
        net = SAC(state_dim,action_dim,max_action)
        net.load(checkpoint)
        net.eval_mode = True
        eval_policy(net, env_name, seed, is_render=True)
    elif args.algorithm == 'DDPG':
        net = DDPG(state_dim, action_dim, max_action)
        net.load(checkpoint)
        eval_policy(net, env_name, seed, is_render=True)
    elif args.algorithm == 'DQN':
        net = DQN(state_dim, action_dim)
        net.load(checkpoint)
        net.eval_mode = True
        eval_policy(net, env_name, seed, is_render=True)

