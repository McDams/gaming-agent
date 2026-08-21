"""Environnement Snake minimal, jouable avec ou sans rendu graphique (PyGame)."""
import random
from collections import namedtuple
from enum import Enum

import numpy as np

BLOCK_SIZE = 20
GRID_W, GRID_H = 20, 15  # cases -> fenêtre 400x300
SPEED = 40  # FPS en mode rendu

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# ordre horaire utilisé pour tourner à droite/gauche
_CLOCKWISE = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


class SnakeEnv:
    """Action = 0 (tout droit), 1 (tourner à droite), 2 (tourner à gauche)."""

    def __init__(self, render=False, max_steps_without_food=100):
        self.render_enabled = render
        self.max_steps_without_food = max_steps_without_food
        self._screen = None
        self._clock = None
        self._font = None
        if self.render_enabled:
            self._init_render()
        self.reset()

    def _init_render(self):
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
        pygame.display.set_caption("Snake - Gaming Agent")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 20)

    def reset(self):
        self.direction = Direction.RIGHT
        cx, cy = GRID_W // 2, GRID_H // 2
        self.head = Point(cx, cy)
        self.snake = [self.head, Point(cx - 1, cy), Point(cx - 2, cy)]
        self.score = 0
        self.steps_since_food = 0
        self.frame = 0
        self._place_food()
        return self.get_state()

    def _place_food(self):
        while True:
            p = Point(random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
            if p not in self.snake:
                self.food = p
                return

    def step(self, action):
        """action in {0,1,2}. Retourne (state, reward, done, score)."""
        self.frame += 1

        if self.render_enabled:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        if self._is_collision() or self.steps_since_food > self.max_steps_without_food:
            done = True
            reward = -10
            if self.render_enabled:
                self._draw()
            return self.get_state(), reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1

        if self.render_enabled:
            self._draw()
            self._clock.tick(SPEED)

        return self.get_state(), reward, done, self.score

    def _is_collision(self, point=None):
        point = point or self.head
        if point.x < 0 or point.x >= GRID_W or point.y < 0 or point.y >= GRID_H:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        idx = _CLOCKWISE.index(self.direction)
        if action == 0:
            new_dir = _CLOCKWISE[idx]
        elif action == 1:
            new_dir = _CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = _CLOCKWISE[(idx - 1) % 4]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        self.head = Point(x, y)

    def _point_ahead(self, direction, steps):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += steps
        elif direction == Direction.LEFT:
            x -= steps
        elif direction == Direction.DOWN:
            y += steps
        elif direction == Direction.UP:
            y -= steps
        return Point(x, y)

    def get_state(self):
        """État discret : 14 booléens (dangers à 1 case, dangers à 2 cases, direction,
        position nourriture). Les dangers à 2 cases donnent à l'agent un avertissement
        avant d'être coincé dans une impasse, ce que le danger à 1 case seul ne permet pas."""
        head = self.head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        idx = _CLOCKWISE.index(self.direction)
        straight_dir = _CLOCKWISE[idx]
        right_dir = _CLOCKWISE[(idx + 1) % 4]
        left_dir = _CLOCKWISE[(idx - 1) % 4]

        state = [
            # danger tout droit / à droite / à gauche, à 1 case
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # danger tout droit / à droite / à gauche, à 2 cases (avertissement anticipé)
            self._is_collision(self._point_ahead(straight_dir, 2)),
            self._is_collision(self._point_ahead(right_dir, 2)),
            self._is_collision(self._point_ahead(left_dir, 2)),
            # direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # position de la nourriture
            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def _draw(self):
        pygame = self._pygame
        self._screen.fill((0, 0, 0))
        for p in self.snake:
            pygame.draw.rect(
                self._screen, (0, 200, 0), (p.x * BLOCK_SIZE, p.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            )
        pygame.draw.rect(
            self._screen,
            (200, 0, 0),
            (self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
        )
        text = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        self._screen.blit(text, (5, 5))
        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            self._pygame.quit()
