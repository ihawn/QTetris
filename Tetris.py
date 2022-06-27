import pygame
import numpy as np
import random
from enum import Enum

pygame.init()
pygame.font.init()

WIDTH = 10
HEIGHT = 30
BLOCK_SIZE = 40
SPEED = 6
TOP_BUFFER = 2

COLORS = {
    1: (255, 255, 0),
    2: (0, 213, 255),
    3: (255, 0, 0),
    4: (0, 255, 0),
    5: (255, 162, 0),
    6: (0, 0, 255),
    7: (162, 0, 255)
}


class Direction(Enum):
    RIGHT = [1, 0]
    LEFT = [-1, 0]
    DOWN = [0, 1]


class Game:

    def __init__(self, w=WIDTH*BLOCK_SIZE, h=HEIGHT*BLOCK_SIZE):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        self.game_matrix = np.zeros((WIDTH, HEIGHT))
        self.piece_dispenser = PieceDispenser()
        self.score = 0
        self.reset()
        self.current_piece = None
        self.current_piece_block_positions = []

    def run_game_step(self):

        self.get_human_input()

        if self.current_piece is None:
            self.current_piece = self.piece_dispenser.get_random_piece()
        else:
            self.move_piece(Direction.DOWN)

        self.clear_row_if_available()

        if self.is_game_over():
            self.reset()

        self.draw_frame()
        self.clock.tick(SPEED)

    def move_piece(self, direction):

        offset = direction.value
        if not self.piece_collision(offset, self.current_piece):
            self.clear_current_piece()
            self.current_piece.x += offset[0]
            self.current_piece.y += offset[1]
            self.draw_current_piece()

        elif direction == direction.DOWN:
            self.current_piece = None  # spawn next piece

    def rotate_piece(self, direction):
        self.clear_current_piece()

        rot = self.current_piece.orientation
        old_rot = rot
        rot += 1 if direction == Direction.RIGHT else -1
        if rot >= len(self.current_piece.piece_positions):
            rot = 0
        elif rot < 0:
            rot = len(self.current_piece.piece_positions) - 1

        self.current_piece.orientation = rot
        self.current_piece.piece_array = self.current_piece.piece_positions[rot]

        can_rotate = True
        if self.piece_collision([0, 0], self.current_piece):  # if there's a collision, determine how to shift piece
            rng = [x for x in range(-1, 2) if x != 0]         # in order to rotate it
            can_rotate = False
            for n in rng:
                for m in rng:
                    if not self.piece_collision([n, m], self.current_piece):  # found an orientation that works
                        self.current_piece.x += n
                        self.current_piece.y += m
                        can_rotate = True
                        break
                else:
                    continue
                break

        if not can_rotate:
            self.current_piece.orientation = old_rot
            self.current_piece.piece_array = self.current_piece.piece_positions[old_rot]

        self.draw_current_piece()

    def clear_current_piece(self):
        for i in range(len(self.current_piece.piece_array)):
            for j in range(len(self.current_piece.piece_array[0])):
                if self.current_piece.piece_array[i][j] != 0:
                    self.game_matrix[i + self.current_piece.x][j + self.current_piece.y] = 0

    def draw_current_piece(self):
        for i in range(len(self.current_piece.piece_array)):
            for j in range(len(self.current_piece.piece_array[0])):
                block = self.current_piece.piece_array[i][j]
                if block != 0:
                    self.game_matrix[i + self.current_piece.x][j + self.current_piece.y] = block

    def piece_collision(self, offset, piece):  # offset is a 1x2 array containing move info
        current_piece_block_positions = self.get_block_positions_of_current_piece()
        for i in range(len(piece.piece_array)):
            for j in range(len(piece.piece_array[0])):
                if piece.piece_array[i][j] != 0:  # not empty spot in piece
                    x = piece.x + offset[0] + i
                    y = piece.y + offset[1] + j
                    if (x >= WIDTH or x < 0 or  # wall
                        y >= HEIGHT or  # floor

                       (self.game_matrix[x][y] != 0 and  # other piece below
                        [x, y] not in current_piece_block_positions)):  # don't check collision against current piece
                        return True
        return False

    def clear_row_if_available(self):
        current_piece_block_positions = self.get_block_positions_of_current_piece()
        for y in reversed(range(HEIGHT)):
            block_count = 0
            for x in range(WIDTH):
                if self.game_matrix[x][y] == 0 or [x, y] in current_piece_block_positions:  # can't clear if current
                    continue                                                                # piece isn't set
                block_count += 1
            if block_count == WIDTH:  # row was cleared
                self.score += 10
                for x in range(WIDTH):  # zero out row
                    self.game_matrix[x][y] = 0
                for y2 in reversed(range(y)):  # move everything above down one
                    for x2 in range(WIDTH):
                        self.game_matrix[x2][y2 + 1] = self.game_matrix[x2][y2]

    def draw_frame(self):
        pygame.event.get()
        self.display.fill((0, 0, 0))
        for i in range(WIDTH):
            for j in range(0, HEIGHT):
                if self.game_matrix[i, j] != 0:
                    self.draw_block(i, j, self.game_matrix[i, j])
        pygame.draw.line(self.display, (255, 255, 255), (0, TOP_BUFFER*BLOCK_SIZE), (WIDTH * BLOCK_SIZE, TOP_BUFFER*BLOCK_SIZE), 2)

        font = pygame.font.SysFont('Courier New', 40)
        score_surface = font.render('Score: ' + str(self.score), False, (255, 255, 255))
        self.display.blit(score_surface, (5, 5))

        pygame.display.flip()

    def draw_block(self, x, y, id):
        pygame.draw.rect(self.display, COLORS[id], pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE*0.95, BLOCK_SIZE*0.95))

    def get_block_positions_of_current_piece(self):
        positions = []
        if self.current_piece is None:
            return positions
        for i in range(len(self.current_piece.piece_array)):
            for j in range(len(self.current_piece.piece_array[0])):
                if self.current_piece.piece_array[i][j] != 0:
                    positions.append([self.current_piece.x + i, self.current_piece.y + j])
        return positions

    def is_game_over(self):
        current_piece_block_positions = self.get_block_positions_of_current_piece()
        for y in range(TOP_BUFFER):
            for x in range(WIDTH):
                if self.game_matrix[x][y] != 0 and [x, y] not in current_piece_block_positions:
                    return True
        return False

    def get_human_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN and self.current_piece is not None:
                if event.key == pygame.K_LEFT:
                    self.move_piece(Direction.LEFT)
                if event.key == pygame.K_RIGHT:
                    self.move_piece(Direction.RIGHT)
                if event.key == pygame.K_q:
                    self.rotate_piece(Direction.LEFT)
                if event.key == pygame.K_e:
                    self.rotate_piece(Direction.RIGHT)

    def reset(self):
        self.frames = 0
        self.score = 0
        for i in range(WIDTH):
            for j in range(HEIGHT):
                self.game_matrix[i][j] = 0


class PieceDispenser:

    def __init__(self):
        self.pieces = np.array([
            [
                [[1, 1],
                 [1, 1]]
            ],

            [
                [[2],
                 [2],
                 [2],
                 [2]],

                [[2, 2, 2, 2]]
            ],

            [
                [[3, 0],
                 [3, 3],
                 [0, 3]],

                [[0, 3, 3],
                 [3, 3, 0]]
            ],

            [
                [[0, 4],
                 [4, 4],
                 [4, 0]],

                [[4, 4, 0],
                 [0, 4, 4]]
            ],

            [
                [[5, 5],
                 [0, 5],
                 [0, 5]],

                [[0, 0, 5],
                 [5, 5, 5]],

                [[5, 0],
                 [5, 0],
                 [5, 5]],

                [[5, 5, 5],
                 [5, 0, 0]]
            ],

            [
                [[0, 6],
                 [0, 6],
                 [6, 6]],

                [[6, 0, 0],
                 [6, 6, 6]],

                [[6, 6],
                 [6, 0],
                 [6, 0]],

                [[6, 6, 6],
                 [0, 0, 6]]
            ],

            [
                [[0, 7],
                 [7, 7],
                 [0, 7]],

                [[0, 7, 0],
                 [7, 7, 7]],

                [[7, 0],
                 [7, 7],
                 [7, 0]],

                [[7, 7, 7],
                 [0, 7, 0]]
            ]
        ], dtype=object)

    def get_random_piece(self):
        id = random.randint(0, 6)
        return Piece(self.pieces[id], id + 1)


class Piece:

    def __init__(self, piece_positions, id):
        self.piece_positions = piece_positions
        self.piece_array = piece_positions[0]
        self.x = WIDTH // 2
        self.y = 0
        self.id = id
        self.orientation = 0


if __name__ == '__main__':
    gm = Game()
    while True:
        gm.run_game_step()
