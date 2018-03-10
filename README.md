# pixelcopter-DQN
Train agent to play pixelcopter (from PyGame Learning Environment) from pixel data using Deep Q-Learning.
***
## Beginning:
![New agent](https://media.giphy.com/media/1wqoTQ7grLkduRDKvF/giphy.gif)

## After 30 minutes of training:
![30 minutes](https://media.giphy.com/media/1yMzoLaVRm2DoOwLLN/giphy.gif)

## Current Issue: Unstable and oscillating policy
Implemented (inspired by [DeepMind Atari paper](https://www.nature.com/articles/nature14236 "Human-level control through deep reinforcement learning")):

- Error clipping [-1, 1]

- Reward clipping [-1. 1]

- Separate target network: Update every 10 training periods
