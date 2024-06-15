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
        self.direction= Vector2(1,0)
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
        self.direction = Vector2(1, 0)


class MAIN():

    def __init__(self):
         self.snake = Snake()
         self.fruit = Fruit()
         self.score = 0



    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()

    def make_elements(self):
        self.draw_grass()
        self.fruit.make_fruit()
        self.snake.make_snake()



    def check_collision(self):
            if self.fruit.pos == self.snake.body[0] :
                self.fruit.random_pos()
                self.snake.add_block()
                self.snake.play_crunch_sound()
                self.score +=1

            for block in self.snake.body[1:]:
                if block == self.fruit.pos:
                    self.fruit.random_pos()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < grid_no or not 0 <= self.snake.body[0].y < grid_no:
            self.game_over()
            return True
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()
                return True
        return False

    def is_collision(self, point):
        if point.x < 0 or point.x >= grid_no or point.y < 0 or point.y >= grid_no:
            return True
        if point in self.snake.body:
            return True
        return False

    def game_over(self):
        self.snake.reset()
        self.score = 0

    def show_score(self):
        score_text = f'Score: {self.score}'
        score_surface = game_font.render(score_text, True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(grid_no * grid_size // 2, grid_no * grid_size - 20))
        screen.blit(score_surface, score_rect)

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



    def play_step(self, action):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        self.snake_move(action)
        self.update()

        screen.fill((255, 128, 0))
        self.make_elements()
        self.show_score()
        pygame.display.update()
        clock.tick(60)
        reward = self.get_reward()
        done = self.check_fail()
        score = self.score
        return reward, done, score

    def snake_move(self, action):
        clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        if self.snake.direction not in clock_wise:
            return

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
            return -10  # Penalty for dying
        if self.fruit.pos == self.snake.body[0]:
            return 10  # Reward for eating food

            # Calculate distance to food before and after moving
        head = self.snake.body[0]
        distance_before = np.linalg.norm(np.array([head.x, head.y]) - np.array([self.fruit.pos.x, self.fruit.pos.y]))

        # Assuming self.snake.direction is updated before calling get_reward()
        new_head = head + self.snake.direction
        distance_after = np.linalg.norm(
            np.array([new_head.x, new_head.y]) - np.array([self.fruit.pos.x, self.fruit.pos.y]))

        # Reward for moving closer to food, penalty for moving away
        if distance_after < distance_before:
            return 1  # Moving closer to food
        else:
            return -1  # Moving away from food



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


