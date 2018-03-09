# %%
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, binarize
from pygame.constants import K_w, K_s
%matplotlib inline
#os.environ["SDL_VIDEODRIVER"] = 'none' #'dummy'

# %%
memory, epsilon = None, None
memory = deque(maxlen=10000)
gamma = 0.95    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.996
learning_rate = 0.005
minibatch_size = 128
img_size = 48

print("Building the model")
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(img_size, img_size, 4)))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (2, 2), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(2))
adam = Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adam)
print("Finished")
model.summary()


act_dict_decode = {0:0,1:K_w}
act_dict_encode = {v: k for k, v in act_dict_decode.items()}

def process(img):
    img = binarize(img)
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    return img

def act(state):
    if np.random.rand() <= epsilon:
        # The agent acts randomly
        return act_dict_decode[np.random.randint(2)]
    # Pick the action based on the predicted reward
    return act_dict_decode[np.argmax(model.predict(state))]

EPISODES = 20000
reward_discount = 0.99
time_per_episode = 10000

game = Pixelcopter(img_size,img_size)
env = PLE(game)
action_size = 2
score = np.zeros(EPISODES//10)
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
        if done: reward = 0
        memory.append([state, action, reward, next_state, done]) # save to memory
        state = next_state
        if done: break

    print("episode: {}/{}: {} timesteps".format(e+1, EPISODES, time+1))

    if e % 10 == 0: score[e // 10] = time+1

    if time + 1 > max_score:
        max_score = time + 1
        model.save('pixelcopter-ai.h5')

    if time == time_per_episode: break

    if len(memory) > minibatch_size and (e+1) % 5 == 0 and e > 450:
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
            targets[i] = model.predict(state_)
            Q_sa = model.predict(next_state_)

            targets [i, action_] = reward_ + gamma * np.amax(Q_sa)

        model.fit(inputs,targets,epochs=1,verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


# %% Play using trained agent
model = load_model('pixelcopter-ai.h5')
act_dict_decode = {0:0,1:K_w}
act_dict_encode = {v: k for k, v in act_dict_decode.items()}
memory = deque(maxlen=4)

def act(state):
    return act_dict_decode[np.argmax(model.predict(state))]

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
        if reward >= 0: reward = 1
        if done: reward = 0
        next_state = process(env.getScreenGrayscale()) #@ next state
        done = env.game_over() # check game over and reassign reward
        memory.append([state, action, reward, next_state, done]) # save to memory
        state = next_state
        if done: break
        #if e % 10 == 0:
            #score[e//10] = time
    #for t in range(-2,-time_per_episode-1,-1):
    #    memory[t][2] =  memory[t][2] + reward_discount * memory[t+1][2]
    print("episode: {}/{}: {} timesteps"
          .format(e+1, 20, time+1))

memory = None
