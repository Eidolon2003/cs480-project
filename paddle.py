from turtle import Turtle

MOVEMENT = 70


class Mover(Turtle):
    def __init__(self):
        super().__init__()
        self.color('white')
        self.shape('square')
        self.penup()
        self.shapesize(stretch_wid=1,stretch_len=10)
        self.goto(x=0,y=-280)
    def moveLeft(self):
        self.backward(MOVEMENT)
    def moveRight(self):
        self.forward(MOVEMENT)


