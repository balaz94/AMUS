import numpy as np
from random import randrange
from keras.models import load_model

class AgentDDQN():
    def __init__(self, gamma, actions_count, model1, model2, steps,
                 experience_replay = None, epsilon = 1.0, epsilon_dec = 1e-5,
                 epsilon_min = 0.01, batch_size = 64, name = 'model'):
        self.gamma = gamma
        self.actions_count = actions_count
        self.model1 = model1
        self.model2 = model2

        self.update_steps = steps
        self.current_steps = 0

        self.is_model1 = True

        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.experience_replay = experience_replay

        self.name = name

    def choose_action(self, state):
        r = np.random.random()

        if r < self.epsilon:
            action = randrange(self.actions_count)
            return action
        else:
            state = state[np.newaxis, :]
            if self.is_model1:
                actions = self.model1.predict(state)
            else:
                actions = self.model2.predict(state)
            return np.argmax(actions)

    def remember(self, state, action, reward, state_, terminal):
        self.experience_replay.store(state, action, reward, state_, terminal)

    def learn(self):
        if self.experience_replay.index < 100:
            return

        if self.is_model1:
            self.model1, self.model2 = self.__fit(self.model1, self.model2)
        else:
            self.moedl2, self.model1 = self.__fit(self.model2, self.model1)

        self.current_steps += 1
        if self.current_steps == self.update_steps:
            self.is_model1 = not self.is_model1
            self.current_steps = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_dec

    def __fit(self, online_model, target_model):
        states, actions, rewards, states_, terminals = self.experience_replay.sample(self.batch_size)
        q_y = online_model.predict(states)
        q_next = online_model.predict(states_)
        q_evaluated = target_model.predict(states_)

        for i in range(0, len(states)):
            q_y[i, actions[i]] = rewards[i] + self.gamma * q_evaluated[i, np.argmax(q_next[i])] * (1 - terminals[i])

        online_model.fit(states, q_y, verbose=0)
        return online_model, target_model

    def save_model(self):
        self.model.save(self.name + '_online.h5')
        self.target_model.save(self.name + '_target.h5')

    def load_model(self):
        self.model = load_model(self.name + '_online.h5')
        self.target_model = load_model(self.name + '_target.h5')
