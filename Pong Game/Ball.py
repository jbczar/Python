import random
import Platform

import main


class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.x_speed = random.choice([-1, 1]) * main.BALL_X_SPEED
        self.y_speed = random.choice([-1, 1]) * main.BALL_Y_SPEED

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

        # Odbicie od górnej i dolnej ściany
        if self.y - self.radius <= 0 or self.y + self.radius >= main.HEIGHT:
            self.y_speed *= -1

    def reset(self):
        self.x = main.WIDTH // 2
        self.y = main.HEIGHT // 2
        self.x_speed = random.choice([-1, 1]) * main.BALL_X_SPEED
        self.y_speed = random.choice([-1, 1]) * main.BALL_Y_SPEED
