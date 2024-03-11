import pygame

import random
import Ball
import Platform

# Stałe
WIDTH = 800
HEIGHT = 600
PLATFORM_WIDTH = 10
PLATFORM_HEIGHT = 80
BALL_RADIUS = 10
PLATFORM_SPEED = 5
BALL_X_SPEED = 6
BALL_Y_SPEED = 6
FPS = 60
WINNING_SCORE = 5


BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
LIGTHGREY=(192,192,192)
BLACK = (13,13,13)


pygame.init()
pygame.font.init()


WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")


SCORE_FONT = pygame.font.SysFont("ARIAL", 40)
MENU_FONT = pygame.font.SysFont("ARIAL", 40)


LOGO_IMAGE = pygame.image.load("Pong.jpg")
LOGO_IMAGE = pygame.transform.scale(LOGO_IMAGE, (300, 200))


def draw(window, paddles, ball, left_score, right_score):
    window.fill(BLUE)

    #rysowanie planszy i wynikow
    pygame.draw.rect(window, RED, paddles[0].rect)
    pygame.draw.rect(window, RED, paddles[1].rect)
    pygame.draw.circle(window, WHITE, (ball.x, ball.y), ball.radius)
    pygame.draw.line(window, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 1)


    left_score_text = SCORE_FONT.render(str(left_score), 1, WHITE)
    right_score_text = SCORE_FONT.render(str(right_score), 1, WHITE)
    window.blit(left_score_text, (WIDTH // 4, 10))
    window.blit(right_score_text, (WIDTH * 3 // 4 - right_score_text.get_width(), 10))

    pygame.display.update()


def handle_paddle_movement(keys, left_paddle, right_paddle):
    if keys[pygame.K_w]:
        left_paddle.move_up()
    if keys[pygame.K_s]:
        left_paddle.move_down()
    if keys[pygame.K_UP]:
        right_paddle.move_up()
    if keys[pygame.K_DOWN]:
        right_paddle.move_down()


def handle_collision(ball, left_paddle, right_paddle):
    if ball.x - ball.radius <= left_paddle.rect.x + left_paddle.rect.width and left_paddle.rect.y <= ball.y <= left_paddle.rect.y + left_paddle.rect.height:
        ball.x_speed *= -1
    elif ball.x + ball.radius >= right_paddle.rect.x and right_paddle.rect.y <= ball.y <= right_paddle.rect.y + right_paddle.rect.height:
        ball.x_speed *= -1


def show_menu():
    run = True

    while run:
        WIN.fill(BLACK)

        logo_rect = LOGO_IMAGE.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        WIN.blit(LOGO_IMAGE, logo_rect)


        instruction_text = MENU_FONT.render("Press SPACE to start",1, WHITE)
        rules_text = MENU_FONT.render("Player 1: WS | Player 2: Up/Down arrows", 1, WHITE)



        instruction_text_rect = instruction_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 120))

        rules_text_rect = rules_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 180))


        WIN.blit(instruction_text, instruction_text_rect)
        WIN.blit(rules_text, rules_text_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    run = False


def show_endgame_menu():
    run = True

    while run:
        WIN.fill(BLACK)

        title_text = MENU_FONT.render("Game Over", 1, WHITE)
        instruction_text = MENU_FONT.render("Press R to play again or Q to quit", 1, WHITE)
        title_text_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
        instruction_text_rect = instruction_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))

        WIN.blit(title_text, title_text_rect)
        WIN.blit(instruction_text, instruction_text_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    run = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()


def main():
    show_menu()

    run = True
    clock = pygame.time.Clock()

    # Tworzenie paletki i piłki
    left_paddle = Platform.Platform(10, HEIGHT // 2 - PLATFORM_HEIGHT // 2, PLATFORM_WIDTH, PLATFORM_HEIGHT)
    right_paddle = Platform.Platform(WIDTH - 10 - PLATFORM_WIDTH, HEIGHT // 2 - PLATFORM_HEIGHT // 2, PLATFORM_WIDTH, PLATFORM_HEIGHT)
    ball = Ball.Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

    left_score = 0
    right_score = 0

    while run:
        clock.tick(FPS)
        draw(WIN, [left_paddle, right_paddle], ball, left_score, right_score)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        keys = pygame.key.get_pressed()
        handle_paddle_movement(keys, left_paddle, right_paddle)

        ball.move()
        handle_collision(ball, left_paddle, right_paddle)

        if ball.x < 0:
            right_score += 1
            ball.reset()
        elif ball.x > WIDTH:
            left_score += 1
            ball.reset()

        won = False
        if left_score >= WINNING_SCORE:
            won = True
            win_text = "Left Player Won!"
        elif right_score >= WINNING_SCORE:
            won = True
            win_text = "Right Player Won!"

        if won:
            show_endgame_menu()
            ball.reset()
            left_paddle.reset()
            right_paddle.reset()
            left_score = 0
            right_score = 0

    pygame.quit()


if __name__ == '__main__':
    main()
