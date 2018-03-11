# pixelcopter-DQN
Train agent to play pixelcopter (from PyGame Learning Environment) from pixel data using Deep Q-Learning.

***
## Beginning
![New agent](https://media.giphy.com/media/1wqoTQ7grLkduRDKvF/giphy.gif)

## After 1.5 hour of learning
Agent awares of boundaries and obstacles, although fine movement control can definitely use some improvement.

![1 hour](https://media.giphy.com/media/WxlxVAhRNFdKhwyQfb/giphy.gif) ![1 hour](https://media.giphy.com/media/6276HOopTYk81YfX30/giphy.gif)

***
## Current Issue: Unstable and oscillating policy
The following has been implemented (inspired by [DeepMind Atari paper](https://www.nature.com/articles/nature14236 "Human-level control through deep reinforcement learning")):

- Error clipping [-1, 1]

- Reward clipping [-1, 1]

- Separate target network: Update every 10 training periods

Implementing these had definitely helped stabilizing Q-function, however oscillation is still observed.
