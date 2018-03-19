# pixelcopter-DQN
Self-taught AI agent learn to play pixelcopter (PyGame Learning Environment) from raw pixel data using Deep Q-Learning.

### Dynamic velocity
Pixelcopter proves to be an interesting to solve iteratively from raw pixel data: The agent needs to be aware of the **compounding effect of action on velocity** (i.e. velocity of the "pixel" varies based on both current velocity and the action taken). To help the agent recognize the effect, the problem is setup similarly to DeepMind Atari paper where input images have 4 channels for 4 most recent frames. The difference: Here every other frame is chosen from the latest frame, equivalent to "seeing" 8 frames into the "past". This seems to accelerate the rate at which policy improves and speeds up learning.  
  
  
***
## Beginning
Agent chooses random actions. (~10 frames/episode on average)

![New agent](https://media.giphy.com/media/1wqoTQ7grLkduRDKvF/giphy.gif)

## After 1.5 hour of training
Agent awares of boundaries and obstacles and realizes the benefits of avoiding them. Fine movement control can definitely use some improvement. (~150 frames/episode on average)

![1 hour](https://media.giphy.com/media/WxlxVAhRNFdKhwyQfb/giphy.gif) ![1 hour](https://media.giphy.com/media/6276HOopTYk81YfX30/giphy.gif)

## After 24 hour (1 million frames) of training 
Agent chooses more appropriate actions and seems to become aware of the compounding effect of action choice on velocity. (~300 frames/episode on average)

![24 hour](https://media.giphy.com/media/3scDuVXdFmG6TYX34V/giphy.gif)

***
## Current Issue: Unstable and oscillating policy
The following has been implemented (inspired by [DeepMind Atari paper](https://www.nature.com/articles/nature14236 "Human-level control through deep reinforcement learning")):

- Error clipping [-1, 1]

- Reward clipping [-1, 1]

- Separate target network: Update every 10 training periods

Implementing these had definitely helped stabilizing Q-function, however oscillation is still observed.
