import pygame,sys,random
from pygame.math import Vector2
import numpy as np


class Fruit():
    def __init__(self):
        self.random_pos()

    def random_pos(self):
        self.x = random.randint(0, grid_no - 1)
        self.y = random.randint(0, grid_no - 1)
        self.pos = Vector2(self.x, self.y)

    def make_fruit(self):
        x_pos = int(self.pos.x * grid_size)
        y_pos = int(self.pos.y * grid_size)
        fruit_rect = pygame.Rect(x_pos, y_pos, grid_size, grid_size)
        screen.blit(fruit_img, fruit_rect)

class Snake():
    def __init__(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction= Vector2(0,0)
        self.new_block = False
        self.crunch_sound = pygame.mixer.Sound('Sound/crunch.mp3')

    def make_snake(self):
        for block in self.body:
            pygame.draw.rect(screen, (0,0,153), pygame.Rect(block.x*grid_size, block.y*grid_size, grid_size, grid_size))
            pygame.draw.rect(screen, (0,0,204), pygame.Rect(block.x*grid_size + 4, block.y*grid_size + 4, 20, 20))
            pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(block.x * grid_size + 8, block.y * grid_size + 8, 10, 10))

    def move_snake(self):
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0,body_copy[0]+self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def play_crunch_sound(self):
            self.crunch_sound.play()

    def reset(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(0, 0)


class MAIN():
    def __init__(self):
         self.snake = Snake()
         self.fruit = Fruit()

    def update(self):
        self.snake.move_snake()
        self.fruit_snake_col()
        self.snake_wall_col()

    def make_elements(self):
        self.draw_grass()
        self.fruit.make_fruit()
        self.snake.make_snake()

        self.score()

    def check_collision(self):
            if self.fruit.pos == self.snake.body[0] :
                self.fruit.random_pos()
                self.snake.add_block()
                self.snake.play_crunch_sound()

            for block in self.snake.body[1:]:
                if block == self.fruit.pos:
                    self.fruit.random_pos()

    def check_fail(self):

        def check_fail(self):
            if not 0 <= self.snake.body[0].x < grid_no or not 0 <= self.snake.body[0].y < grid_no:
                return True
            for block in self.snake.body[1:]:
                if block == self.snake.body[0]:
                    return True
            return False

    def game_over(self):
        self.snake.reset()

    def draw_grass(self):
        grass_color = (244,123,0)
        for row in range(grid_no):
            if row % 2 == 0:
                for col in range(grid_no):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * grid_size, row * grid_size, grid_size, grid_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
            else:
                for col in range(grid_no):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * grid_size, row * grid_size, grid_size, grid_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)

    def score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render("Score: " + score_text, True, (56, 74, 12))
        score_x = int(grid_size * grid_no/2)
        score_y = int(20)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        fruit_rect = fruit_img.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(fruit_rect.left, fruit_rect.top, fruit_rect.width + fruit_rect.width + 50,
                              fruit_rect.height)

        pygame.draw.rect(screen, (0,204,204), bg_rect)
        screen.blit(score_surface, score_rect)
        screen.blit(fruit_img, fruit_rect)
        pygame.draw.rect(screen, (56, 74, 12), bg_rect, 2)

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.snake_move(action)
        self.update()

        screen.fill((255, 128, 0))
        self.make_elements()
        pygame.display.update()
        clock.tick(60)
        reward = self.get_reward()
        done = self.check_fail()
        score = len(self.snake.body) - 3
        return reward, done, score

    def snake_move(self, action):
        clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.snake.direction = new_dir

    def get_reward(self):
        if self.check_fail():
            return -10
        if self.fruit.pos == self.snake.body[0]:
            return 10
        return 0

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
grid_size = 30
grid_no = 20
screen = pygame.display.set_mode((grid_no * grid_size, grid_no * grid_size))
clock = pygame.time.Clock()
fruit_img = pygame.image.load('images/foodproject.png').convert_alpha()
game_font = pygame.font.Font(None, 25)
bg_music = pygame.mixer.Sound('Sound/music.wav')
bg_music.play(loops=-1)
bg_music.set_volume(0.1)


