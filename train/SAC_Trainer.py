import torch
import gym
import numpy as np

from utils.utils import ReplayBuffer
from eval.eval import eval_policy
from workers.SAC_worker import SAC

def SAC_Trainer(logger, saver, cfg_dict, checkpoint):
    # init
    state_dimention = cfg_dict.get('state_dimention', 17)
    action_dimention = cfg_dict.get('action_dimention', 6)
    max_action_value = cfg_dict.get('max_action_value', 1)
    seed = cfg_dict.get('seed', 0)
    env_name = cfg_dict.get('env_name')
    start_timesteps = cfg_dict.get('start_timesteps', 25000)
    eval_freq = cfg_dict.get('eval_freq', 5000)
    max_timesteps = cfg_dict.get('max_timesteps', 1000000)
    batch_size = cfg_dict.get('batch_size', 256)
    discount = cfg_dict.get('discount', 0.99)
    tau = cfg_dict.get('tau', 0.005)
    lr = cfg_dict.get('lr', 3e-4)
    log_std_max = cfg_dict.get('log_std_max', 2)
    log_std_min = cfg_dict.get('log_std_min', -20)
    alpha = cfg_dict.get('alpha', 0.2)
    automatic_entropy_tuning = cfg_dict.get('automatic_entropy_tuning', 0)

    model = SAC(state_dimention, action_dimention, max_action_value, lr, discount, tau, log_std_max,
                log_std_min, alpha, automatic_entropy_tuning)

    if checkpoint  is not None:
        model.load(checkpoint)

    # Set seeds
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer(state_dimention, action_dimention, cfg_dict.get('capacity'))

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
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = model.select_action(state)

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
        if (t + 1) % eval_freq == 0:
            model.eval_mode = True
            evaluations.append(eval_policy(model, env_name, seed))
            saver.save_evaluation(evaluations)
            saver.save_checkpoint(model.save(), reward=evaluations[-1])
            model.eval_mode = False
