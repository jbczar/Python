import pygame

import main
import Ball


class Platform:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
    def move_up(self):
        if self.rect.y > 0:
            self.rect.y -= main.PLATFORM_SPEED

    def move_down(self):
        if self.rect.y < main.HEIGHT - self.rect.height:
            self.rect.y += main.PLATFORM_SPEED

    def reset(self):
        self.rect.y = main.HEIGHT // 2 - self.rect.height // 2
