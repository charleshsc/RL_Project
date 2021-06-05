#Reinforcement Learning

This repository contains Pytorch implementation of RL.

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