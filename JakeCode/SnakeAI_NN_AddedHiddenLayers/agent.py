import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, plt

import sys

MAX_MEMORY = 100000
BATCH_SIZE = 1000

LR = 0.1              # 0.001, 0.1
GAMMA = 0.9             # 0.9, 0.4
nHiddenLayers = 1       # 1, 2, 3
layer1Nodes = 250        # 250, 25, 100
layer2Nodes = 0        # 100, 100, 100
layer3Nodes = 0        # 25, 250, 100

imageText = '1H_250_LRp1_Gp9.png'


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0        #controls randomness
        self.gamma = GAMMA      #discount rate (<1)
        self.memory = deque(maxlen=MAX_MEMORY)  #popleft()
        #self.model = Linear_QNet(11,layer1Nodes,layer2Nodes,layer3Nodes,3)      #over-fitting with 3 hidden layers
        #self.model = Linear_QNet(11,layer1Nodes,layer2Nodes,3)                  #2 hidden layers
        self.model = Linear_QNet(11,layer1Nodes,3)                              #1 hidden layers
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            #danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            game.food.x < game.head.x,  #food left
            game.food.x > game.head.x,  #food right
            game.food.y < game.head.y,  #food up
            game.food.y > game.head.y  #food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, acion, reward, next_state, done):
        self.memory.append((state, acion, reward, next_state, done)) #one tuple, popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        #as number of games increases, random moves decrease
        self.epsilon = 150 - self.n_games   #after 150 games, no more randomness
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    over30 = 0
    over30_nGames = 999
    over50 = 0
    over50_nGames = 999
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get current state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (replay memory), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

            # record # of games over 30 pts
            if score >= 30:
                over30 = over30+1
                if over30 == 3:
                    over30_nGames = agent.n_games

            # record # of games over 50 pts
            if score >= 50:
                over50 = over50+1
                if over50 == 3:
                    over50_nGames = agent.n_games

            # stop agent after 400 games
            if agent.n_games >= 400:
                print('Simulation Ended at 400')
                print('Total # of Games with Score >= 30pts: ', over30)
                print('# of Game to 3rd Score >= 30pts: ', over30_nGames)
                print('Total # of Games with Score >= 30pts: ', over50)
                print('# of Game to 3rd Score >= 30pts: ', over50_nGames)

                plt.savefig(imageText)
                
                #save data to csv file
                data = np.asarray([[LR, GAMMA, nHiddenLayers, layer1Nodes, layer2Nodes, layer3Nodes, mean_score, record, over30, over30_nGames, over50, over50_nGames]])

                file_path = "data.csv"
                with open(file_path, 'ab') as f:  # 'ab' is for append binary mode
                    np.savetxt(f, data, delimiter=",")

                raise EndRun("400th Game")
                #exit()

                
class EndRun(Exception):
    pass



if __name__ == '__main__':
    #train()
    while True:
        try:
            train()
        except Exception as e:
            print(f"Resetting: {e}")
            continue
        else:
            #print("Execution completed successfully.")
            break

# Re-execute the script
python = sys.executable
os.execl(python, python, * sys.argv)
    

