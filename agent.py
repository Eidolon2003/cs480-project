import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from game import PLAY
import numpy as np
from collections import deque
import random
import turtle as tr


class NETWORK(nn.Module):
    def __init__(self, n_obv, n_action):
        super().__init__()
        self.linear1 = nn.Linear(n_obv, 256)
        self.linear2 = nn.Linear(256, n_action)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        x = self.linear2(x)

        return x

    def save(self, file_name='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)


class MODEL:
    def __init__(self, model):
        self.epsilon = 1.0
        self.decay_rate = 0.005
        self.learning_rate = 0.9
        self.discount_rate = 0.8
        self.model = model
        self.optimize = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.learning_rate * torch.max(self.model(nextState[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimize.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimize.step()


class PLAYER:

    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.discount_rate = 0.9
        self.mem = deque(maxlen=10000)
        self.model = NETWORK(7, 3)
        self.trained = MODEL(self.model)

    def state(self, game):
        paddle = game.player
        ball = game.objective
        brick = game.bricks

        state = [
            #  paddle middle
            # paddle left
            # paddle right
            (0 < paddle.xcor() < 400),
            (400 < paddle.xcor() < 800),
            (800 < paddle.xcor() < 1200),

            ball.xcor() > paddle.xcor(),
            ball.xcor() < paddle.xcor(),
            ball.xcor() == paddle.xcor(),
            len(brick.bricks),

            # ball to left of paddle
            # ball to right of paddle
            # brick layout
            # amount of bricks
        ]
        return np.array(state, dtype=int)

    def learn(self, state, move, reward, next_action, done):
        self.mem.append((state, move, reward, next_action, done))

    def long_mem(self):
        if len(self.mem) > 1000:
            long_sample = random.sample(self.mem, 1000)
        else:
            long_sample = self.mem
        thisState, thisAction, thisReward, thisNextState, thisDone = zip(*long_sample)
        self.trained.trainStep(thisState, thisAction, thisReward, thisNextState, thisDone)

    def short_mem(self, action, state, reward, nextState, done):
        self.trained.trainStep(state, action, reward, nextState, done)

    def chooseAction(self, state):
        self.epsilon = max(0, 80 - self.n_game)

        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            nextState = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(nextState)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


# add more randomness
#
def train():
    scores = []
    mean_score = []
    total = 0
    highest = 0
    agent = PLAYER()
    game = PLAY()

    while True:
        oldState = agent.state(game)
        move = agent.chooseAction(oldState)
        reward, done, score = game.playStep(move)
        newState = agent.state(game)

        agent.short_mem(move, oldState, reward, newState, done)
        agent.learn(oldState, move, reward, newState, done)

        if done:

            game.reset()
            agent.n_game += 1
            agent.long_mem()

            if score.score > highest:
                highest = score.score
                agent.model.save()
            print('Game: ', agent.n_game, ' Score: ', reward, ' Record: ', highest)

            scores.append(score.score)
            total += score.score
            mean_score_temp = total / agent.n_game
            mean_score.append(mean_score_temp)
            if agent.n_game % 100 == 0:
                print(f'Mean score: {mean_score_temp}')
        done = False


if __name__ == '__main__':
    train()
