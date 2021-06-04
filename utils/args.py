import argparse
import glob
import os.path as osp
import os

def obtain_env_args():
    abs_dir = osp.realpath(".")  # 当前的绝对位置
    root_name = 'code'
    root_dir = abs_dir[:abs_dir.index(root_name) + len(root_name)]
    directory = osp.join(root_dir, 'run')

    runs = sorted(glob.glob(osp.join(directory, 'experiment_*')))
    run_id = max([int(x.split('_')[-1]) for x in runs]) + 1 if runs else 0
    if run_id != 0 and len(os.listdir(osp.join(directory, 'experiment_{}'.format(str(run_id - 1))))) == 0:
        run_id = run_id - 1
    experiment_dir = osp.join(directory, 'experiment_{}'.format(str(run_id)))

    if not osp.exists(experiment_dir):
        os.makedirs(experiment_dir)

    save = experiment_dir

    env_choices = [
        "Humanoid-v2",
        "HalfCheetah-v2",
        "VideoPinball-ramNoFrameskip-v4",
        "Ant-v2",
        "Hopper-v2"
    ]

    algorithm_choices = [
        'A3C',
        'TD3',
        'SAC',
        'DDPG',
        'DQN'
    ]

    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument('--env_name', type=str, default="VideoPinball-ramNoFrameskip-v4", choices=env_choices)
    parser.add_argument('--save', type=str, default=save)
    parser.add_argument('--checkpoint',type=str,default=None)
    parser.add_argument('--algorithm',type=str,default='DQN', choices=algorithm_choices)

    args = parser.parse_args()
    args.env_name = os.path.join(root_dir,'config',args.env_name)
    return args

args = obtain_env_args()