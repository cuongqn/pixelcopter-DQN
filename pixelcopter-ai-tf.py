# %%
import os
import math
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
from collections import deque
from sklearn.preprocessing import MinMaxScaler, binarize
from pygame.constants import K_w, K_s
%matplotlib inline
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("poster")
#os.environ["SDL_VIDEODRIVER"] = 'dummy' #'none' or 'dummy'


# %%
# Constants
memory, epsilon = None, None
memory = deque(maxlen=50000)
gamma = 0.95    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.998
learning_rate = 0.0008
minibatch_size = 128
img_size = 80

# Build model
# Variables: Xaiver init

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 4])
y_true = tf.placeholder(tf.float32, shape=[None, 2])
input_layer = tf.reshape(x, [-1, img_size, img_size, 4])

conv1 = tf.layers.conv2d(inputs=input_layer,
                         filters=32,
                         kernel_size=[8, 8],
                         strides=[4, 4],
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)


conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=64,
                         kernel_size=[4, 4],
                         strides=[2, 2],
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)

conv3 = tf.layers.conv2d(inputs=conv2,
                         filters=64,
                         kernel_size=[2, 2],
                         strides=[1, 1],
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
conv3_flat = tf.layers.flatten(conv3)

dense4 = tf.layers.dense(inputs=conv3_flat,
                         units=256,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)

y = tf.layers.dense(inputs=dense4,
                    units=2,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

# Loss: MSE
loss = tf.losses.mean_squared_error(labels=y_true,predictions=y)
# Optimizer: Adam or RMSprop
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# %%
act_dict_decode = {0:0,1:K_w}
act_dict_encode = {v: k for k, v in act_dict_decode.items()}
saver = tf.train.Saver()
sess = tf.Session()
sess_frozen = tf.Session()
sess.run(tf.global_variables_initializer())
sess_frozen = sess

def process(img):
    img = binarize(img)
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    return img

def act(state):
    if np.random.rand() <= epsilon:
        # The agent acts randomly
        return act_dict_decode[np.random.randint(2)]
    action_predict = sess.run(y, {x: state})
    # Pick the action based on the predicted reward
    return act_dict_decode[np.argmax(action_predict)]

EPISODES = 6000
OBSERVATIONS = 300
#reward_discount = 0.99
time_per_episode = 1000
game = Pixelcopter(img_size,img_size)
env = PLE(game)
action_size = 2
score_mean = np.zeros(EPISODES//10)
score_std = np.zeros(EPISODES//10)
score_last10 = []
training_count = 0
plt.figure()
max_score = 0

for e in range(EPISODES):
    env.init()
    state = process(env.getScreenGrayscale())
    for time in range(time_per_episode):
        # Set actions
        if time < 3:
            action = act_dict_decode[0]
        else:
            action_input = np.concatenate((state,
                                           memory[-1][0],
                                           memory[-2][0],
                                           memory[-3][0]), axis=3)
            action = act(action_input)

        reward = env.act(action) # get reward from action
        next_state = process(env.getScreenGrayscale()) #@ next state
        done = env.game_over() # check game over and reassign reward
        if reward >= 0: reward = 1
        if reward < 0: reward = 0
        if done: reward = 0
        memory.append([state, action, reward, next_state, done]) # save to memory
        state = next_state
        if done: break

    # Uncomment to print score every episode
    print("episode: {}/{}: {} timesteps"
          .format(e+1, EPISODES, time+1))

    if (e+1) % 10 != 0: score_last10.append(time+1)
    if (e+1) % 10 == 0:
        score_mean[e // 10] = np.mean(score_last10)
        score_std[e // 10] = np.std(score_last10)

        if np.amax(score_last10) > max_score:
            max_score = np.amax(score_last10)
            saver.save(sess, './pixelcopter-ai-tf-')

        del score_last10[:]

        ## Uncomment to plot score every 10 episodes
        # plt.plot(score_mean[:(e // 10)])
        # plt.fill_between(np.arange(e // 10),
        #                  score_mean[:(e // 10)]+score_std[:(e // 10)],
        #                  score_mean[:(e // 10)]-score_std[:(e // 10)],
        #                  alpha=0.5)
        #
        # plt.show()

    if time == time_per_episode: break

    if len(memory) > minibatch_size and (e+1) % 10 == 0 and e > OBSERVATIONS:
        training_count += 1
        if training_count % 10 == 0:
            sess_frozen = sess
        minibatch_index = np.random.randint(3,len(memory)-1,
                                            size=minibatch_size)
        minibatch = random.sample(memory, minibatch_size)
        inputs = np.zeros((minibatch_size, state.shape[1], state.shape[2], 4))
        targets = np.zeros((minibatch_size, action_size))
        for i, index in enumerate(minibatch_index):
            state_ = np.concatenate((memory[index][0],
                                     memory[index-1][0],
                                     memory[index-2][0],
                                     memory[index-3][0]), axis=3)
            action_ = act_dict_encode[memory[index][1]]
            reward_ = memory[index][2]
            next_state_ = np.concatenate((memory[index][3],
                                          memory[index-1][3],
                                          memory[index-2][3],
                                          memory[index-3][3]), axis=3)

            inputs[i] = state_
            targets[i] = sess_frozen.run(y, {x: state_})
            Q_sa = sess_frozen.run(y, {x: next_state_})

            targets [i, action_] = reward_ + gamma * np.amax(Q_sa)

        sess.run(optimizer,feed_dict = {x: inputs, y_true: targets})

        ## Sinusoidal epsilon decay
        # epsilon = (epsilon_decay**training_count)/2*(1+math.cos(2*math.pi*training_count*8/(EPISODES//5)))

        # Exponential epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


# %% Play using trained agent
sess_ = tf.Session()
saver = tf.train.Saver()
saver.restore(sess_, "pixelcopter-ai-tf-")
act_dict_decode = {0:0,1:K_w}
act_dict_encode = {v: k for k, v in act_dict_decode.items()}
memory = deque(maxlen=4)

def act(state):
    action_predict = sess_.run(y, {x: state})
    # Pick the action based on the predicted reward
    return act_dict_decode[np.argmax(action_predict)]

def process(img):
    img = binarize(img)
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    return img

game = Pixelcopter(img_size,img_size)
env = PLE(game,fps=30,display_screen=True)

for e in range(20):
    env.init()
    state = process(env.getScreenGrayscale())
    for time in range(5000):
        # Set actions
        if time < 3:
            action = act_dict_decode[0]
        else:
            action_input = np.concatenate((state,
                                           memory[-1][0],
                                           memory[-2][0],
                                           memory[-3][0]), axis=3)
            action = act(action_input)

        reward = env.act(action) # get reward from action
        next_state = process(env.getScreenGrayscale()) #@ next state
        done = env.game_over() # check game over and reassign reward
        memory.append([state, action, reward, next_state, done]) # save to memory
        state = next_state
        if done: break

    print("episode: {}/{}: {} timesteps"
          .format(e+1, 20, time+1))

memory = None
