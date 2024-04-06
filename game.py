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
        self.AI = Agent()

        self.score = Scoreboard(lives=5)


        self.player = Mover()
        self.objective = Hitter()
        self.bricks = Wall()

        self.inProgress = True


        self.screen.listen()
        self.screen.onkey(key='Left', fun=player.moveLeft)
        self.screen.onkey(key='Right', fun=player.moveRight)

    def reset(self):
        self.score.reset()
        self.player.reset()
        self.objective.reset()
        self.bricks.reset()
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

            action = self.MODEL.chooseAction(state=state)

            if action == 0:
                self.player
            elif action == 1:
                self.player.moveLeft()
            elif action == 2:
                self.player.moveRight()

            self.objective.fly()
            self.blockHit()
            self.paddleHit()
            self.hasHit()
            if len(bricks.bricks) == 0:
                self.UI.gameOver(win=True)
                break

        def hasHit():
                global objective, inProgress

                if objective.xcor() < -580 or objective.xcor() > 570:
                    objective.hit(xHit=True, yHit=False)
                    return

                if objective.ycor() > 270:
                    objective.hit(xHit=False, yHit=True)
                    return

                if objective.ycor() < -280:

                    self.objective.reset()
                    self.score.decreaseLives()
                    if self.score.lives == 0:
                        self.score.reset()
                        self.UI.gameOver(False)
                        inProgress = False
                        return

                    return

        def paddleHit():
                global objective, player
                playerX = player.xcor()
                objectiveX = objective.xcor()

                if objective.distance(player) < 110 and objective.ycor() < -250:
                    if playerX > 0:
                        if objectiveX > playerX:
                            objective.hit(xHit=True, yHit=True)
                            return
                        else:
                            objective.hit(xHit=False, yHit=True)
                            return

                    elif playerX < 0:
                        if objectiveX < playerX:
                            objective.hit(xHit=True, yHit=True)
                            return
                        else:
                            objective.hit(xHit=False, yHit=True)
                            return


                    else:
                        if objectiveX > playerX:
                            objective.hit(xHit=True, yHit=True)
                            return
                        elif objectiveX < playerX:
                            objective.hit(xHit=True, yHit=True)
                            return
                        else:
                            objective.hit(xHit=False, yHit=True)
                            return

        def blockHit():
                global objective, bricks

                for brick in bricks.bricks:
                    if objective.distance(brick) < 40:
                        self.score.increaseScore()
                        brick.quantity -= 1
                        if brick.quantity == 0:
                            brick.clear()
                            brick.goto(3000, 3000)
                            bricks.bricks.remove(brick)
                        if objective.xcor() < brick.left_wall:
                            objective.hit(xHit=True, yHit=False)
                        elif objective.xcor() > brick.right_wall:
                            objective.hit(xHit=True, yHit=False)
                        elif objective.ycor() < brick.bottom_wall:
                            objective.hit(xHit=False, yHit=True)
                        elif objective.ycor() > brick.upper_wall:
                            objective.hit(xHit=False, yHit=True)















play = game()
game.playStep()
tr.mainloop()
