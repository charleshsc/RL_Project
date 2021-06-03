import gym
import numpy as np


def eval_policy(policy, env_name, seed, eval_episodes=10, is_render=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    if is_render:
        eval_env.render()

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            if is_render:
                eval_env.render()
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
