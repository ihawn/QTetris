import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from enum import Enum
from collections import deque


BATCH_SIZE = 2000
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.9
TRAIN_POOL_SIZE = 100000


class TetrisAI:

    def __init__(self, input_size, game):
        self.probability_curve_divisor = 3000
        self.training_pool = deque(maxlen=TRAIN_POOL_SIZE)
        self.model = Model(input_size, 256, 256, 5)  # left, stay, right, rotate left, rotate right
        self.q_step = QStep(self.model, learning_rate=LEARNING_RATE, discount_rate=DISCOUNT_RATE, tetris_ai=self)
        self.game = game

    def get_state_vector(self):

        mtx = self.game.game_matrix
        game_state = np.zeros(16 + len(mtx) * len(mtx[0]))

        if self.game.current_piece is not None:
            arr = self.game.current_piece.piece_array
            for i in range(4):
                k = 0
                for j in range(4*i, 4*(i+1)):
                    if i < len(arr) and k < len(arr[0]) and arr[i][k] != 0:
                        game_state[j] = 1
                    k += 1

        for i in range(len(mtx)):
            k = 0
            for j in range(len(mtx[0])*i, len(mtx[0])*(i+1)):
                if mtx[i][k] != 0:
                    game_state[j + 16] = 1
                k += 1

        return np.array(game_state, dtype=int)

    def train(self, current_state, next_state, action, reward, game_over):

        self.q_step.train_step(current_state, next_state, action, reward, game_over)
        self.training_pool.append((current_state, next_state, action, reward, game_over))

        if game_over:
            if len(self.training_pool) > BATCH_SIZE:
                batch = random.sample(self.training_pool, BATCH_SIZE)
            else:
                batch = self.training_pool

            states, next_states, actions, rewards, game_overs = zip(*batch)
            self.q_step.train_step(states, next_states, actions, rewards, game_overs)


class Model(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

        self.conv1 = nn.Sequential(nn.Linear(input_size, hidden_size_1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(hidden_size_1, hidden_size_2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(hidden_size_2, output_size))

    def forward(self, input):
        # input = self.linear1(input)
        # input = F.relu(input)
        # input = self.linear2(input)
        # input = F.relu(input)
        # input = self.linear3(input)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        return input


class QStep:

    def __init__(self, model, learning_rate, discount_rate, tetris_ai):
        self.model = model
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.tetris_ai = tetris_ai

    def get_ai_input(self):

        random_move_probability = math.exp(-self.tetris_ai.game.game_count / self.tetris_ai.probability_curve_divisor)

        move_id = 0
        if random.uniform(0, 1) < random_move_probability:
            move_id = random.randint(0, 4)
        else:
            game_state = self.tetris_ai.get_state_vector()
            tensor_state = torch.tensor(game_state, dtype=torch.float)
            prediction = self.model(tensor_state)
            move_id = torch.argmax(prediction).item()

        return move_id

    def train_step(self, current_states, next_states, actions, rewards, game_over):
        current_states = np.array(current_states)
        next_states = np.array(next_states)
        current_states = torch.tensor(current_states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)

        if len(current_states.shape) == 1:
            current_states = torch.unsqueeze(current_states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            game_over = (game_over, )

        predictions = self.model(current_states)
        targets = predictions.clone()

        for i in range(len(rewards)):
            q_value = rewards[i]
            if not game_over[i]:
                max_outome = torch.max(self.model(next_states[i]))
                q_value += self.discount_rate * max_outome

            max_action_index = torch.argmax(actions[i]).item()
            targets[i][max_action_index] = q_value

        self.optimizer.zero_grad()
        loss = self.criterion(targets, predictions)
        loss.backward()
        self.optimizer.step()
