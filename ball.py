from turtle import Turtle

MOVEMENT = 10


class Hitter(Turtle):
    def __init__(self):
        super().__init__()
        self.shape('circle')
        self.color('red')
        self.penup()
        self.xMove = MOVEMENT
        self.yMove = MOVEMENT
        self.reset()

    def fly(self):
        new_y = self.ycor() + self.yMove
        new_x = self.xcor() + self.xMove
        self.goto(new_x, new_y)


    def nextPos(self):
        new_y = self.ycor() + self.yMove
        new_x = self.xcor() + self.xMove
        return new_x, new_y



    def hit(self, xHit, yHit):
        if xHit:
            self.xMove *= -1

        if yHit:
            self.yMove *= -1

    def reset(self):
        self.goto(x=0, y=-240)
        self.yMove = 10
