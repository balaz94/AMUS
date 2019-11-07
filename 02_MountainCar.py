import gym
import time
import datetime
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from algorithms.ddqn_tau import AgentDDQN
from algorithms.experience_replay import ExperienceReplay

def build_network():
    model = Sequential()
    model.add(Dense(16, input_shape=(2, )))
    model.add(Activation('relu'))
    model.add(Dense(16, input_shape=(2, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

def learning():
    env = gym.make('MountainCar-v0')
    games_count = 1001
    scores = []
    input = 2
    actions = 3
    buffer = ExperienceReplay(10000, (input, ))

    d_now = datetime.datetime.now()
    agent = AgentDDQN(0.99, actions, build_network(), build_network(), 0.01,
                 buffer, batch_size = 64,
                 epsilon = 1.0, epsilon_min = 0.05, epsilon_dec = 0.001,
                 name = 'models/02_MountainCar_DDQN_16_16')

    for i in range(1, games_count):
        score = 0
        terminal = False
        state = env.reset()

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal, info = env.step(action)
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
    env = gym.make('MountainCar-v0')

    input = 2
    actions = 3

    agent = AgentDDQN(0.99, actions, build_network(), build_network(), 0.01,
                 epsilon = 0, name = 'models/02_MountainCar_DDQN_16_16')
    agent.load_model()

    while True:
        done = False
        observation = env.reset()

        while not done:
            env.render()
            time.sleep(0.01)
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_

    env.close()

if __name__ == '__main__':
    learning()
