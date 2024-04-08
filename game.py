import turtle as tr
import torch
import breaks
from agent import Agent, MODEL, NETWORK
from paddle import Mover
from ball import Hitter
from breaks import Wall
from score import Scoreboard
from ui import ui
import time




class game:
    def __init__(self):

        self.screen = tr.Screen()





        self.screen.setup(width=1200, height=600)
        self.screen.bgcolor('black')
        self.screen.tracer(0)

        self.UI = ui()
        self.UI.text()
        #self.MODEL = MODEL()

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
        self.bricks.reset()
        self.reward = 0
        self.inProgress = True

    def playStep(self):
        while self.inProgress:
            self.screen.update()
            time.sleep(.01)
            state = {
                'agentXCor': self.player.xcor(),
                'objXCor': self.objective.xcor(),
                'objYCor': self.objective.ycor(),
                'brickQuan': len(self.bricks.bricks),
                'brickLayout': [(brick.xcor(), brick.ycor()) for brick in self.bricks.bricks]
            }

            #action = self.MODEL.chooseAction(state=state)

            # if action == 0:
            #     self.player
            # elif action == 1:
            #     self.player.moveLeft()
            # elif action == 2:
            #     self.player.moveRight()

            self.objective.fly()
            self.blockHit()
            self.paddleHit()
            self.hasHit()
            if len(self.bricks.bricks) == 0:
                self.UI.gameOver(win=True)
                break


    def hasHit(self):
            global objective, inProgress

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
        global objective, player
        playerX = self.player.xcor()
        self.objectiveX =self.objective.xcor()

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
            global objective, bricks

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


play = game()
play.playStep()
tr.mainloop()
