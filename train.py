import yaml
import os
import time

from utils.logger import get_logger
from utils.args import obtain_env_args
from utils.Saver import Saver

from train import *

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
        TD3_Trainer(logger, saver, cfg_dict.get('TD3'), checkpoint)
    elif args.algorithm == 'SAC':
        SAC_Trainer(logger, saver, cfg_dict.get('SAC'), checkpoint)
    elif args.algorithm == 'DDPG':
        DDPG_Trainer(logger, saver, cfg_dict.get('DDPG'), checkpoint)


