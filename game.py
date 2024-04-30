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
        tr.speed("fastest")
        self.score = Scoreboard(lives=5)

        self.reward = 0
        self.player = Mover()
        self.objective = Hitter()
        self.bricks = Wall()
        self.objectiveX = self.objective.xcor()

        self.inProgress = False

        self.screen.listen()

    def reset(self):
        self.score.reset()
        self.player.reset()
        self.objective.reset()
        self.bricks.reset()
        self.UI.reset()
        self.reward = 0


    def playStep(self, action):
        done = False
        self._move(action)
        self.objective.fly()
        self.blockHit()
        self.paddleHit()
        self.hasHit()
        self.screen.update()
        if len(self.bricks.bricks) == 0:
            self.reward += 500
        done = len(self.bricks.bricks) == 0 or self.score.lives == 0
        time.sleep(0.0001)


        return self.reward, done, self.score




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
                self.reward -= 100

                if self.score.lives == 0:
                    self.reward -= 150
                    self.UI.gameOver(False)
                    self.inProgress = False

            if self.player.xcor() < -580:
                self.player.goto(-560, self.player.ycor())
            elif self.player.xcor() > 570:
                self.player.goto(550, self.player.ycor())

            return

    def paddleHit(self):

        playerX = self.player.xcor()
        self.objectiveX = self.objective.xcor()

        if self.objective.distance(self.player) < 110 and self.objective.ycor() < -250:
            # Check specific collision areas on the paddle
            paddle_left = playerX - 50
            paddle_right = playerX + 50

            if self.objectiveX >= paddle_left and self.objectiveX <= paddle_right:

                self.objective.hit(xHit=False, yHit=True)
                self.reward += 50
                self.objective.sety(self.objective.ycor() + 15)
                return
            elif self.objectiveX < paddle_left:

                self.objective.hit(xHit=True, yHit=True)
                self.reward += 50
                self.objective.sety(self.objective.ycor() + 15)
                return
            elif self.objectiveX > paddle_right:

                self.objective.hit(xHit=True, yHit=True)
                self.reward += 50
                self.objective.sety(self.objective.ycor() + 15)
                return

        return

    def blockHit(self):

        for brick in self.bricks.bricks[:]:
            if self.objective.distance(brick) < 40:
                self.score.increaseScore()
                brick.quantity -= 1
                self.reward += 10
                if brick.quantity == 0:
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
        return

    def _move(self, move):
        if move == [0, 1, 0] and self.player.xcor() >= -560:
            self.player.moveLeft()
        elif move == [0,0,1] and self.player.xcor() <= 550:
            self.player.moveRight()
        else:
            pass



