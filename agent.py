import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os
import numpy as np
from collections import deque

import random

class NETWORK(nn.Module):
        def __init__(self, n_obv, n_action):
            super().__init__()
            self.linear1 = nn.Linear(n_obv, 256)
            self.linear2 = nn.Linear(256, n_action)

        def forward(self, train):
            train = self.relu(self.linear1(train))
            train = self.linear2(train)
            return train

class MODEL:
        def __init__(self, model):
            self.epsilon = 1.0
            self.decay_rate = 0.005
            self.learning_rate = 0.9
            self.discount_rate = 0.8
            self.prevAction = None
            self.prevState = None
            self.model = model
            self.optimize = optim.Adam(model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()





        def trainStep(self, state, action, reward, nextState, done):
            state = torch.tensor(state, dtype=torch.float)
            nextState = torch.tensor(nextState, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            if len(state.shape) == 1:
                     state = torch.unsqueeze(state, 0)
                     nextState = torch.unsqueeze(nextState, 0)
                     action = torch.unsqueeze(nextState, 0)
                     reward = torch.unsqueeze(nextState, 0)
                     done = (done, )

            prediction = self.model(state)
            ob = prediction.clone()
            for x in range(len(done)):
                Q = reward[x]
                if not done[x]:
                    Q = reward[x] + self.discount_rate * torch.max(self.model(nextState[x]))

                ob[x][torch.argmax(action[x]).item()] = Q

            self.optimize.zero_grad()
            lose = self.criterion(ob, prediction)
            lose.backward()

            self.optimize.step()



class Agent:



    def __init__(self):
        self.game = 0
        self.epsilon = 0
        self.discount_rate = 0.9
        self.mem = deque(maxlen=100000)
        self.model = NETWORK(10, 3)
        self.trained = MODEL(self.model)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            stateTensor = torch.tensor(state, dtype=torch.float)
            qValues = self.mod.model(stateTensor)
            action = torch.argmax(qValues).item()
        return action
