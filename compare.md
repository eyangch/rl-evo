## Comparing Algorithms

Each of these algorithms, reinforcement learning and has its pros and cons, but what do they look like in action?

Here is my implementation reinforcement learning playing a simple [cartpole](https://gym.openai.com/envs/CartPole-v1/) game:

![RL Cartpole](./rl.gif)

At first, it is complete and utter garbage at the game. It keeps falling over without any "strategy". After just 30 seconds, the agent learns some strategies, such as countering the falling pole by moving in that direction.

<details>
  <summary>Reinforcement Learning Cartpole Code (Click to expand)</summary>
  
  ```py
import os
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from matplotlib import animation
import matplotlib.pyplot as plt
 
def save_frames_as_gif(frames, path='./', filename='rl.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='ffmpeg', fps=60)

def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
    model.summary()
    return model
 
frames = []

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 30
        self.memory = deque(maxlen=2000)
 
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = self.batch_size
 
        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
         
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))
 
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(state, target ,batch_size=self.batch_size, verbose=0)
 
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                frames.append(self.env.render(mode="rgb_array"))
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                self.replay()
 
if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    save_frames_as_gif(frames)
```
    
</details>

Cartpole is a rather simple game, so it can quickly become rather good at the game. However, a more complicated game, like snake, can take much longer, for days, or even weeks, to become proficient at it. Here is a reinforcement algorithm to play snake I implemented with tensorflow and pygame.
