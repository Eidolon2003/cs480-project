from turtle import Turtle

try:
    score= int(open('highest_score.txt', 'r').read())
except FileNotFoundError:
    score=open('highest_score.txt','w').write(str(0))
except ValueError:
    score = 0

Font = ('arial', 18, 'normal')

class Scoreboard(Turtle):
    def __init__(self, lives):
        super().__init__()
        self.color('white')
        self.penup()
        self.hideturtle()
        self.Highscore = score
        self.goto(x=-580,y=260)
        self.lives = lives
        self.score = 0
        self.updateScore()

    def updateScore(self):
        self.clear()
        self.write(f"Score: {self.score} | Highest Score: {self.Highscore} | "
                   f"Lives: {self.lives}", align='left', font=Font)

    def increaseScore(self):
        self.score+= 1
        if self.score > self.Highscore:
            self.Highscore += 1
        self.updateScore()

    def decreaseLives(self):
        self.lives -= 1
        self.updateScore()

    def reset(self):
        self.clear()
        self.score = 0
        self.updateScore()
        open('highest_score.txt', 'w').write(str(self.Highscore))
