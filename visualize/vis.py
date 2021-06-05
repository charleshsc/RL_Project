import numpy as np
import matplotlib.pyplot as plt

def plt_evaluations(name):
    evaluations = np.load(name)
    num_all_steps = range(len(evaluations))
    plt.plot(num_all_steps, evaluations)
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Reward per Episode")

    plt.show()

if __name__ == '__main__':
    plt_evaluations('../run/TD3_HalfCheetah/evaluation.npy')
