import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from game import PLAY
import numpy as np
from collections import deque
import random



class NETWORK(nn.Module):
    def __init__(self, n_obv, n_action):
        super().__init__()
        self.linear1 = nn.Linear(n_obv, 256)
        self.linear2 = nn.Linear(256, n_action)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x= self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self):
        name = os.path.join('./model', 'model.pth')
        self.load_state_dict(torch.load(name))



class MODEL:
    def __init__(self, model):
        self.learningRate = 0.001
        self.discountRate = 0.9
        self.model = model
        self.optimize = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)


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
        for x in range(len(done)):
            Q_new = reward[x]
            if not done[x]:
                Q_new = reward[x] + self.discountRate * torch.max(self.model(nextState[x]))

            target[x][torch.argmax(action[x]).item()] = Q_new

        self.optimize.zero_grad()

        loss = self.criterion(target, pred)
        loss.backward()
        self.optimize.step()


class PLAYER:

    def __init__(self):
        self.nGame = 0
        self.mem = deque(maxlen=20000)
        self.model = NETWORK(9, 3)
        self.model.load()
        self.trained = MODEL(self.model)

    def state(self, game):
        paddle = game.player
        ball = game.objective
        brick = game.bricks
        brickLayout = [pos for brick in brick.bricks for pos in (brick.xcor(), brick.ycor())]
        newX = ball.xcor() + ball.xMove
        newY = ball.ycor() + ball.yMove
        state = [

            ball.xcor(),
            ball.ycor(),
            newX,
            newY,

            paddle.xcor(),

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
        if len(self.mem) > 64:
            longSample = random.sample(self.mem, 64)
        else:
            longSample = self.mem
        thisState, thisAction, thisReward, thisNextState, thisDone = zip(*longSample)
        self.trained.trainStep(thisState, thisAction, thisReward, thisNextState, thisDone)

    def short_mem(self, action, state, reward, nextState, done):
        self.trained.trainStep(state, action, reward, nextState, done)

    # utilizes simple greedy formula to decide between
    # learned move, and random move,
    # each step
    def chooseAction(self, state, increment):
        # only allow move every other frame, to avoid bot spam
        if increment % 2 == 0:
            self.epsilon = 80 - self.nGame

            action = [0, 0, 0]
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                action[move] = 1
            else:
                # pick, then use move
                nextState = torch.tensor(state, dtype=torch.float)
                prediction = self.model(nextState)
                move = torch.argmax(prediction).item()
                action[move] = 1

            return action
        else:
            action = [0, 0, 0]
            return action


# add more randomness
#
def train():
    scores = []
    meanScore = []
    total = 0
    highest = 0
    agent = PLAYER()
    game = PLAY()
    increment = 0
    tempLives = game.score.lives
    reward = 0

    # set up loop to make game continue forever until terminated
    while True:
        if increment % 20 == 0:
            if tempLives == game.score.lives:
                reward += 100
            tempLives = game.score.lives
        oldState = agent.state(game)
        move = agent.chooseAction(oldState, increment)
        reward, done, score = game.playStep(move)
        newState = agent.state(game)

        agent.short_mem(move, oldState, reward, newState, done)
        agent.learn(oldState, move, reward, newState, done)

        increment += 1
        # done is sent in the event that
        # there are no bricks left, or when lives = 0
        if done:

            increment = 0
            game.reset()

            torch.cuda.empty_cache()

            agent.nGame += 1
            agent.long_mem()


            if reward > highest:
                #agent.model.save()
                highest = reward

            print('Game: ', agent.nGame, ' Score: ', reward, ' Record: ', highest)

            scores.append(reward)
            total += reward
            meanScoreTemp = total / agent.nGame
            meanScore.append(meanScoreTemp)
            if agent.nGame % 100 == 0:
                print(f'Mean score: {meanScoreTemp}')


if __name__ == '__main__':
    train()
