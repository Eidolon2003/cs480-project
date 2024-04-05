import turtle as tr
import torch
import breaks
from paddle import Mover
from ball import Hitter
from breaks import Wall
from score import Scoreboard
from ui import ui
import time



screen = tr.Screen()





screen.setup(width=1200, height=600)
screen.bgcolor('black')
screen.tracer(0)

UI = ui()
UI.text()

score = Scoreboard(lives=5)


player = Mover()
objective = Hitter()
bricks = Wall()

inProgress = True


screen.listen()
screen.onkey(key='Left', fun=player.moveLeft)
screen.onkey(key='Right', fun=player.moveRight)

def hasHit():
        global objective, inProgress

        if objective.xcor() < -580 or objective.xcor() > 570:
            objective.hit(xHit=True, yHit=False)
            return

        if objective.ycor() > 270:
            objective.hit(xHit=False, yHit=True)
            return

        if objective.ycor() < -280:

            objective.reset()
            score.decreaseLives()
            if score.lives == 0:
                score.reset()
                UI.gameOver(False)
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
                score.increaseScore()
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


while inProgress:
    screen.update()
    time.sleep(.01)
    objective.fly()
    blockHit()
    paddleHit()
    hasHit()
    if len(bricks.bricks) == 0:
        UI.gameOver(win=True)
        break












tr.mainloop()

