import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
#font = pygame.font.SysFont('arial', 25)
font = pygame.font.Font('AmaticSC-Regular.ttf',30)

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 40000
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400

# RGB Colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLUE2 = (10,10,150)


class SnakeGameAI:
    def __init__(self, w=WINDOW_WIDTH, h=WINDOW_HEIGHT):

        #game window
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SNAKE GAME :)')
        self.clock = pygame.time.Clock()

        self.reset()
        

    def reset(self):
        #starting position and direction
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)

        #starting snake
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake: #if food was placed inside the snake, re-do
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1

        #collect user input
        self._get_user_input()

        #move
        self._move(action)      #update head
        self.snake.insert(0, self.head) #add new head to start of snake

        #check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #place new food
        if self.head == self.food:  #if snake ate food...
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()    #remove last block of snake

        #update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        #return game over and score
        return reward, game_over, self.score
    

    def _get_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx =  clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]):     #go straight (no change)
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]):   #turn right
            idx = (idx+1)%4
            new_dir = clock_wise[idx]
        else:                                   #turn left
            idx = (idx-1)%4
            new_dir = clock_wise[idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x,y)
    

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        #hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        #hits self
        if pt in self.snake[1:]:
            return True
        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            text = font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0,0])
            pygame.display.flip()




