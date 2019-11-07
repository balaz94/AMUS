import numpy as np
from random import randrange
from keras.models import load_model

class AgentDDQN():
    def __init__(self, gamma, actions_count, model, target_model, tau,
                 experience_replay = None, epsilon = 1.0, epsilon_dec = 1e-5,
                 epsilon_min = 0.01, batch_size = 64, name = 'model'):
        self.gamma = gamma
        self.actions_count = actions_count
        self.online_model = model
        self.target_model = target_model
        self.tau = tau
        self.inv_tau = 1 - tau

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
            actions = self.online_model.predict(state)
            return np.argmax(actions)

    def remember(self, state, action, reward, state_, terminal):
        self.experience_replay.store(state, action, reward, state_, terminal)

    def learn(self):
        if self.experience_replay.index < 100:
            return

        states, actions, rewards, states_, terminals = self.experience_replay.sample(self.batch_size)
        q_y = self.online_model.predict(states)
        q_next = self.online_model.predict(states_)
        q_evaluated = self.target_model.predict(states_)

        for i in range(0, len(states)):
            q_y[i, actions[i]] = rewards[i] + self.gamma * q_evaluated[i, np.argmax(q_next[i])] * (1 - terminals[i])

        self.online_model.fit(states, q_y, verbose=0)

        model_weights = np.array(self.online_model.get_weights()) * self.tau
        target_model_weights = np.array(self.target_model.get_weights()) * self.inv_tau
        update_target_weights = model_weights + target_model_weights
        self.target_model.set_weights(update_target_weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_dec

    def save_model(self):
        self.online_model.save(self.name + '_online.h5')
        self.target_model.save(self.name + '_target.h5')

    def load_model(self):
        self.online_model = load_model(self.name + '_online.h5')
        self.target_model = load_model(self.name + '_target.h5')
