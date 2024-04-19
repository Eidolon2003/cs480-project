from turtle import Turtle
import random

COLOR = ['blue', 'yellow', 'pink', 'purple', 'brown', 'salmon']

POINTS = [1, 2, 1, 1, 3, 2, 1, 4, 1, 3,
           1, 1, 1, 4, 1, 3, 2, 2, 1, 2,
           1, 2, 1, 2, 1]

class Brick(Turtle):
    def __init__(self, xCor, yCor):
        super().__init__()
        self.penup()
        self.shape('square')
        self.shapesize(stretch_wid=1.5,stretch_len=3)
        self.color(random.choice(COLOR))
        self.goto(x=xCor, y=yCor)
        self.quantity = (random.choice(POINTS))

        self.left_wall = self.xcor() - 30
        self.right_wall = self.xcor() + 30
        self.upper_wall = self.ycor() + 15
        self.bottom_wall = self.ycor() - 15


class Wall:
    def __init__(self):

        self.yStart = 0
        self.yEnd = 240
        self.bricks = []
        self.createAllLanes()

    def reset(self):
        self.bricks = []
        self.createAllLanes()

    def createLane(self, yCor):
        for x in range(-570,570,63):
            brick = Brick(x, yCor)
            self.bricks.append(brick)

    def createAllLanes(self):
        for x in range(self.yStart, self.yEnd, 32):
            self.createLane(x)

