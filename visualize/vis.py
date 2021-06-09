import glob
import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

env_name = ["_Ant","_HalfCheetah","_Hopper","_Humanoid"]
algo_list = ["TD3_","DDPG_","SAC_"]

def plt_evaluations(algorithm):
    prefix = '../run/'
    suffix = '/evaluation.npy'
    fig = plt.figure(1,figsize=(12,4))
    gs = gridspec.GridSpec(1,4)  # 设定网格

    for i, env in enumerate(env_name):
        ax = fig.add_subplot(gs[0,i])
        evaluations = np.load(prefix + algorithm + env + suffix)
        num_all_steps = range(len(evaluations))
        plt.plot(num_all_steps, evaluations)
        if i == 0:
            ax.set_xlabel("thousand steps")
            ax.set_ylabel("Average Reward")
        ax.set_title(env[1:]+"-v2")

    fig.suptitle(algorithm)
    plt.show()

def plt_A3C():
    prefix = '../run/'
    suffix = '/evaluation.npy'
    fig = plt.figure(1,figsize=(12,4))
    gs = gridspec.GridSpec(1,4)  # 设定网格
    algorithm = 'A3C'

    for i, env in enumerate(env_name):
        ax = fig.add_subplot(gs[0,i])
        num_all_steps = range(100)

        log_rootpath = prefix + 'A3C' + env
        log_path = glob.glob(os.path.join(log_rootpath,'*.log'))[0]
        reward = []
        num = 0
        with open(log_path, 'r') as f:
            for line in f.readlines():
                if num >= 10000:
                    break
                num += 1
                reward.append(line.split(':')[-1].strip())
        rewards = []
        for i in range(len(reward)):
            if (i + 1) % 100 == 0:
                rewards.append(float(reward[i]))
        plt.plot(num_all_steps, rewards)
        if i == 0:
            ax.set_xlabel("hundred steps")
            ax.set_ylabel("Average Reward")
        ax.set_title(env[1:]+"-v2")

    fig.suptitle(algorithm)
    plt.show()

def plt_env(env_name):
    prefix = '../run/'
    suffix = '/evaluation.npy'
    fig = plt.figure(1,figsize=(12,4))
    gs = gridspec.GridSpec(1,3)

    for i,algo in enumerate(algo_list):
        ax = fig.add_subplot(gs[0, i])
        evaluations = np.load(prefix + algo + env_name + suffix)
        num_all_steps = range(len(evaluations))
        plt.plot(num_all_steps, evaluations)
        if i == 0:
            ax.set_xlabel("Training Epochs")
            ax.set_ylabel("Average Reward per Episode")
        ax.set_title(algo[:-1])

    fig.suptitle(env_name+"-v2")
    plt.show()

def plt_atari():
    fig = plt.figure(1, figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0,0])
    log_path = '../run/DQN_BreakoutNoFrameskip/20210605_155201.log'
    reward = []
    with open(log_path,'r') as f:
        for line in f.readlines():
            reward.append(line.split(':')[-1].strip())
    reward.remove(reward[-1])
    rewards = []
    for i in range(len(reward)):
        if (i+1) % 1000 == 0:
            rewards.append(float(reward[i]))

    x = range(len(rewards))
    plt.plot(x, rewards)
    ax.set_xlabel("thousand steps")
    ax.set_ylabel("Average Reward")
    ax.set_title('BreakoutNoFrameskip')

    ax = fig.add_subplot(gs[0, 1])
    eva_path = '../run/DQN_PongNoFrameskip/evaluation.npy'
    evaluations = np.load(eva_path)
    x = range(len(evaluations))
    plt.plot(x, evaluations)
    ax.set_xlabel("thousand steps")
    ax.set_ylabel("Average Reward")
    ax.set_title('PongNoFrameskip')

    plt.suptitle('DQN')
    plt.show()



if __name__ == '__main__':
    plt_env('Ant')
