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

        def forward(self, train):
            train = F.relu(self.linear1(train))
            train = self.linear2(train)
            return train

        def save(self, file_name='model.pth'):
            model_path ='./model'
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
                     action = torch.unsqueeze(action, 0)
                     reward = torch.unsqueeze(reward, 0)
                     done = (done, )

            prediction = self.model(state)
            ob = prediction.clone()
            for x in range(len(done)):
                Q = reward[x]
                if not done[x]:
                    Q = reward[x] + self.discount_rate * torch.max(self.model(nextState[x]))

                ob[x][action[x].item()] = Q

            self.optimize.zero_grad()
            lose = self.criterion(ob, prediction)
            lose.backward()

            self.optimize.step()

class PLAYER:

    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.discount_rate = 0.9
        self.mem = deque(maxlen=100000)
        self.model = NETWORK(311, 3)
        self.trained = MODEL(self.model)


    def state(self, game):
        paddle = game.player
        ball = game.objective
        brick = game.bricks

        brickLayout = [pos for brick in brick.bricks for pos in (brick.xcor(), brick.ycor())]
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
                *brickLayout

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
        self.epsilon = 80 - self.n_game
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            nextState = torch.tensor(state, dtype=torch.float)
            prediction = self.model(nextState)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train():
    scores = []
    mean_score = []
    total = 0
    highest = 0
    agent = PLAYER()
    game = PLAY()
    i = 0
    while i < 1000:
        game.reset()
        oldState = agent.state(game)
        move = agent.chooseAction(oldState)
        reward, done, score = game.playStep(action=move)
        newState = agent.state(game)

        agent.short_mem(oldState, move, reward, newState, done)

        if done:
            game.reset()
            agent.n_game +=1
            agent.long_mem()

            if score > highest:
                highest = score
            print('Game: ', agent.n_game, ' Score: ', score, ' Record: ', highest)

            scores.append(score)
            total += score
            mean_score_temp = total / agent.n_game
            mean_score.append(mean_score_temp)
            i += 1



if __name__ == '__main__':
    train()

