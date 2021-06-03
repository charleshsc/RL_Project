import torch
import gym
import numpy as np

from utils.utils import ReplayBuffer
from eval.eval import eval_policy
from workers.TD3_worker import TD3

def TD3_Trainer(logger, saver, cfg_dict, checkpoint):
    # init
    state_dimention = cfg_dict.get('state_dimention',17)
    action_dimention = cfg_dict.get('action_dimention',6)
    max_action_value = cfg_dict.get('max_action_value',1)
    seed = cfg_dict.get('seed',0)
    env_name = cfg_dict.get('env_name')
    start_timesteps = cfg_dict.get('start_timesteps',25000)
    max_timesteps = cfg_dict.get('max_timesteps',1000000)
    expl_noise = cfg_dict.get('expl_noise',0.1)
    batch_size = cfg_dict.get('batch_size',256)
    discount = cfg_dict.get('discount',0.99)
    tau = cfg_dict.get('tau',0.005)
    policy_noise = cfg_dict.get('policy_noise',0.2) * max_action_value
    noise_clip = cfg_dict.get('noise_clip',0.5) * max_action_value
    policy_freq = cfg_dict.get('policy_freq',2)

    model = TD3(state_dimention, action_dimention, max_action_value,discount,tau,policy_noise, noise_clip, policy_freq)

    if checkpoint is not None:
        model.load(checkpoint)

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