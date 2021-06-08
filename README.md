#Reinforcement Learning

This repository contains Pytorch implementation of RL.

The code corresponding to the training stage is mainly included in the model, workers, train directories. And
 if you want to add a new algorithm, you can follow these steps:
1. put the main model code into models directory which only contains the class based on the nn.Module.
2. put the agent containing the model in the workers directory. It is important to have `select_action` function and `train` function. In the 
train function, there is no need to act the env and only contains the training data. So if using the replay buffer, the parameters for this function is the 
   `replay_buffer` and `batch_size`.
   
3. put the environment setting and env step in the train directory.
4. Add the algorithm super-parameters in the corresponding env config file which is located in the config directory.

## Algorithm Implemented
1. A3C
2. DDPG
3. DQN
4. TD3
5. SAC

## Environments Implemented
### Hopper-v2
+ Observation space: 8
+ Action space: 3

### Humanoid-v2
+ Observation space: 376
+ Action space: 17

### HalfCheetah-v2
+ Observation space: 17
+ Action space: 6

### Ant-v2
+ Observation space: 111
+ Action space: 8

### VideoPinball-ramNoFrameskip-v4
original observation space: (128,)

Use the atari_wrapper 
+ Observation space: (84, 84, 4)
+ Action space: Discrete(9)
### BreakoutNoFrameskip-v4
original observation space: (210, 160, 3)

Use the atari_wrapper 
+ Observation space: (84, 84, 4)
+ Action space: Discrete(4)
### PongNoFrameskip-v4
original observation space: (210, 160, 3)

Use the atari_wrapper 
+ Observation space: (84, 84, 4)
+ Action space: Discrete(6)
### BoxingNoFrameskip-v4
original observation space: (210, 160, 3)

Use the atari_wrapper 
+ Observation space: (84, 84, 4)
+ Action space: Discrete(18)

## usage
Train the agents on the environments:
```angular2html
python train.py \
        --algorithm TD3 \
        --env_name HalfCheetah-v2 \
        --checkpoint None 
```

Infer the agents on the environments:
```angular2html
python infer.py \
        --algorithm TD3 \
        --env_name HalfCheetah-v2 \
        --checkpoint run/experiment_0/checkpoint.pth
```

## Reference
1. TD3 refers to this [repo](https://github.com/sfujim/TD3)
2. SAC refers to this [repo](https://github.com/dongminlee94/deep_rl/blob/master/agents/sac.py)
3. A3C refers to this [repo](https://github.com/MorvanZhou/pytorch-A3C)