import torch
import gym
import numpy as np

from utils.utils import ReplayBuffer
from eval.eval import eval_policy
from workers.DQN_worker import DQN
from utils.atari_wrapper import wrap_deepmind

def DQN_Trainer(logger, saver, cfg_dict, checkpoint):
    env_name = cfg_dict.get('env_name')
    action_dimention = cfg_dict.get('action_dimention', 6)
    state_dimention = cfg_dict.get('state_dimention', 17)
    capacity = cfg_dict.get('capacity', 1000000)
    batch_size = cfg_dict.get('batch_size', 256)
    seed = cfg_dict.get('seed', 0)
    start_timesteps = cfg_dict.get('start_timesteps', 25000)
    eval_freq = cfg_dict.get('eval_freq', 5000)
    max_timesteps = cfg_dict.get('max_timesteps', 1000000)
    discount = cfg_dict.get('discount', 0.99)
    lr = cfg_dict.get('lr',1e-3)
    epsilon = cfg_dict.get('epsilon', 1)
    epsilon_decay = cfg_dict.get('epsilon_decay', 0.995)
    target_update_step = cfg_dict.get('target_update_step', 100)
    algo = cfg_dict.get('algo', 'DQN')

    model = DQN(state_dimention, action_dimention, discount, epsilon, epsilon_decay, target_update_step, lr, algo)

    if checkpoint is not None:
        model.load(checkpoint)

    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer(state_dimention, action_dimention, capacity)

    model.eval_mode = True
    evaluations = [eval_policy(model, env_name, seed)]
    model.eval_mode = False

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy

        action = model.select_action(np.array(state))

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= 300:
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
        if (t + 1) % eval_freq == 0:
            model.eval_mode = True
            evaluations.append(eval_policy(model, env_name, seed))
            saver.save_evaluation(evaluations)
            saver.save_checkpoint(model.save())
            model.eval_mode = False
