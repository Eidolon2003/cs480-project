import turtle as tr
from enum import Enum
from paddle import Mover
from ball import Hitter
from breaks import Wall
from score import Scoreboard
from ui import ui
import numpy as np
import time



class MOVE(Enum):
    NO = 0
    LEFT = 1
    RIGHT = 2

class PLAY:
    def __init__(self):

        self.screen = tr.Screen()

        self.screen.setup(width=1200, height=600)
        self.screen.bgcolor('black')
        self.screen.tracer(0)

        self.UI = ui()
        self.UI.text()

        self.score = Scoreboard(lives=5)

        self.reward = 0
        self.player = Mover()
        self.objective = Hitter()
        self.bricks = Wall()

        self.inProgress = True

        self.screen.listen()
        self.screen.onkey(key='Left', fun=self.player.moveLeft)
        self.screen.onkey(key='Right', fun=self.player.moveRight)

    def reset(self):
        self.score.reset()
        self.player.reset()
        self.objective.reset()
        self.bricks = Wall()
        self.reward = 0
        self.inProgress = True

    def playStep(self, action):
        self.inProgress =True
        while self.inProgress:

            state = {
                'agentXCor': self.player.xcor(),
                'objXCor': self.objective.xcor(),
                'objYCor': self.objective.ycor(),
                'brickQuan': len(self.bricks.bricks),
                'brickLayout': [(brick.xcor(), brick.ycor()) for brick in self.bricks.bricks]
            }
            print(state)

            self._move(action)

            self.objective.fly()
            self.blockHit()
            self.paddleHit()
            self.hasHit()
            done = len(self.bricks.bricks) == 0 or self.score.lives == 0

            self.screen.update()
            time.sleep(0.01)

            if done:
                self.inProgress = False
                return self.reward, self.inProgress, self.score

        return self.reward, self.inProgress, self.score




    def hasHit(self):

            if self.objective.xcor() < -580 or self.objective.xcor() > 570:
                self.objective.hit(xHit=True, yHit=False)
                return

            if self.objective.ycor() > 270:
                self.objective.hit(xHit=False, yHit=True)
                return

            if self.objective.ycor() < -280:

                self.objective.reset()
                self.score.decreaseLives()
                self.reward -= 50

                if self.score.lives == 0:
                    self.reward -= 250
                    self.score.reset()
                    self.UI.gameOver(False)
                    inProgress = False




            return

    def paddleHit(self):

        playerX = self.player.xcor()
        self.objectiveX = self.objective.xcor()

        if self.objective.distance(self.player) < 110 and self.objective.ycor() < -250:
            if playerX > 0:
                if self.objectiveX > playerX:
                    self.objective.hit(xHit=True, yHit=True)
                    return
                else:
                    self.objective.hit(xHit=False, yHit=True)
                    return

            elif playerX < 0:
                if self.objectiveX < playerX:
                    self.objective.hit(xHit=True, yHit=True)
                    return
                else:
                    self.objective.hit(xHit=False, yHit=True)
                    return


            else:
                if self.objectiveX > playerX:
                    self.objective.hit(xHit=True, yHit=True)
                    return
                elif self.objectiveX < playerX:
                    self.objective.hit(xHit=True, yHit=True)
                    return
                else:
                    self.objective.hit(xHit=False, yHit=True)
                    return

    def blockHit(self):

            for brick in self.bricks.bricks:
                if self.objective.distance(brick) < 40:
                    self.score.increaseScore()
                    brick.quantity -= 1
                    self.reward += 10
                    if brick.quantity == 0:
                        self.reward += 100
                        brick.clear()
                        brick.goto(3000, 3000)
                        self.bricks.bricks.remove(brick)
                    if self.objective.xcor() < brick.left_wall:
                        self.objective.hit(xHit=True, yHit=False)
                    elif self.objective.xcor() > brick.right_wall:
                        self.objective.hit(xHit=True, yHit=False)
                    elif self.objective.ycor() < brick.bottom_wall:
                        self.objective.hit(xHit=False, yHit=True)
                    elif self.objective.ycor() > brick.upper_wall:
                        self.objective.hit(xHit=False, yHit=True)
            return self.reward

    def _move(self, move):
        if np.array_equal(move, [1, 0, 0]):
            pass
        elif np.array_equal(move, [0, 1, 0]):
            self.player.moveLeft()
        else:
            self.player.moveRight()

