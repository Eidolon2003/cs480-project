import time
from turtle import Turtle
import random

font = ("Arial", 52, "normal")
font1 = ("Arial", 32, "normal")
center = "center"
color = "white"
colors = ['blue', 'yellow', 'pink', 'purple', 'brown', 'salmon']

class ui(Turtle):
    def __init__(self):
        super().__init__()
        self.hideturtle()
        self.penup()
        self.color(random.choice(colors))
        self.text()

    def text(self):
        self.clear()
        self.goto(x=0,y=-180)
        self.write(f"Breakout", align=center, font=font1)

    def gameOver(self, win):
        self.clear()
        if win == True:
            self.write("Win!", align=center, font=font)
        else:
            self.write("Lost", align=center, font=font)