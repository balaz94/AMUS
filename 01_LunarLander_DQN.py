import gym
import datetime
import time
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from algorithms.dqn import AgentDQN
from algorithms.experience_replay import ExperienceReplay

def build_dqn(lr, n_actions, input_dims, fc1, fc2):
    model = Sequential()
    model.add(Dense(fc1, input_shape=(input_dims, ), activation='relu'))
    model.add(Dense(fc2, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model

def learning():
    env = gym.make('LunarLander-v2')
    games_count = 1001
    scores = []
    agent = AgentDQN(0.99, 4, build_dqn(0.001, 4, 8, 64, 64),
                     ExperienceReplay(100000, (8, )), update_steps = 2000,
                     batch_size = 64,
                     name = 'models/LunarLander_DQN_64_64.h5')

    d_now = datetime.datetime.now()
    for i in range(1, games_count):
        score = 0
        terminal = False
        state = env.reset()

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal, _ = env.step(action)
            agent.remember(state, action, reward, state_, terminal)
            agent.learn()
            state = state_
            score += reward

        scores.append(score)

        print('episode: ', i, '\t\tscore: ', + score, '\t\taverage score:' , np.average(scores))
        if i % 10 == 0:
            d_end = datetime.datetime.now()
            d = d_end - d_now
            print('time: ', d)
            if i % 100 == 0:
                agent.save_model()

def animation():
    env = gym.make('LunarLander-v2')

    agent = AgentDQN(0.99, 4, build_dqn(0.001, 4, 8, 64, 64),
                 epsilon = 0, name = 'models/LunarLander_DQN_64_64.h5')
    agent.load_model()

    while True:
        terminal = False
        observation = env.reset()

        while not terminal:
            env.render()
            time.sleep(0.01)
            action = agent.choose_action(observation)
            observation, _, terminal, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    learning()
