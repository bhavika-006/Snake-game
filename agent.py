import torch, random
import numpy as np
from pygame.math import Vector2
from collections import deque
import game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.01

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,300,3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self,main_game):
        head = main_game.snake.body[0]
        grid_size = 30
        vector_l = Vector2(head.x - grid_size, head.y)
        vector_r = Vector2(head.x + grid_size, head.y)
        vector_u = Vector2(head.x, head.y - grid_size)
        vector_d = Vector2(head.x, head.y + grid_size)

        dir_l = main_game.snake.direction == Vector2(-1, 0)
        dir_r = main_game.snake.direction == Vector2(1, 0)
        dir_u = main_game.snake.direction == Vector2(0, -1)
        dir_d = main_game.snake.direction == Vector2(0, 1)

        state = [
                # Danger straight
                (dir_r and main_game.is_collision(vector_r)) or
                (dir_l and main_game.is_collision(vector_l)) or
                (dir_u and main_game.is_collision(vector_u)) or
                (dir_d and main_game.is_collision(vector_d)),

                # Danger right
                (dir_u and main_game.is_collision(vector_r)) or
                (dir_d and main_game.is_collision(vector_l)) or
                (dir_l and main_game.is_collision(vector_u)) or
                (dir_r and main_game.is_collision(vector_d)),

                # Danger left
                (dir_d and main_game.is_collision(vector_r)) or
                (dir_u and main_game.is_collision(vector_l)) or
                (dir_r and main_game.is_collision(vector_u)) or
                (dir_l and main_game.is_collision(vector_d)),

                # Move direction
                dir_l,
                dir_r,
                dir_u,
                dir_d,

                # Food location
                main_game.fruit.pos.x < main_game.snake.body[0].x,  # food left
                main_game.fruit.pos.x > main_game.snake.body[0].x,  # food right
                main_game.fruit.pos.y < main_game.snake.body[0].y,  # food up
                main_game.fruit.pos.y > main_game.snake.body[0].y  # food down
                ]
        return torch.tensor(state, dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else: mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self,state):
          self.epsilon = 80-self.n_games
          final_move = [0,0,0] #action
          if random.randint(0,200)< self.epsilon:
              move = random.randint(0,2)
              final_move[move] = 1
          else:
              state0 = torch.tensor(state,dtype = torch.float)
              prediction = self.model(state0)
              move = torch.argmax(prediction).item()
              final_move[move] = 1
          return final_move


def train():

    agent = Agent()
    main_game = game.MAIN()
    state_old = agent.get_state(main_game)

    while True:
        final_move = agent.get_action(state_old)
        reward, done, score = main_game.play_step(final_move)
        state_new = agent.get_state(main_game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        state_old = state_new

        if done:
            # train long memory, plot result
            main_game.game_over()
            agent.n_games += 1
            agent.train_long_memory()




if __name__ == "__main__":
    train()







