import gym
import time
import datetime
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from algorithms.ddqn_tau import AgentDDQN
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
    games_count = 20001
    scores = []

    online_model = build_dqn(0.001, 4, 8, 64, 64)
    target_model = build_dqn(0.001, 4, 8, 64, 64)

    agent = AgentDDQN(0.99, 4, online_model, target_model, 0.01,
                 ExperienceReplay(100000, (8,)), batch_size = 64,
                 epsilon = 0.01, epsilon_min = 0.01, epsilon_dec = 1e-5,
                 name = 'models/LunarLander_DDQN_64_64')

    d_now = datetime.datetime.now()
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
    env = gym.make('LunarLander-v2')

    agent = AgentDDQN(0.99, 4, build_dqn(0.001, 4, 8, 64, 64),
                 build_dqn(0.001, 4, 8, 64, 64), 0.01,
                 epsilon = 0,
                 name = 'models/LunarLander_DDQN_64_64')
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
