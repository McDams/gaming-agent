"""Environnement Snake minimal, jouable avec ou sans rendu graphique (PyGame)."""
import random
from collections import namedtuple
from enum import Enum

import numpy as np

BLOCK_SIZE = 28
GRID_W, GRID_H = 20, 15  # cases -> fenêtre 560x420 (+ bandeau HUD)
HUD_HEIGHT = 56
DEFAULT_SPEED = 12  # FPS en mode rendu, pensé pour être regardable (pas juste fonctionnel)

# Palette
COLOR_BG_A = (24, 28, 36)
COLOR_BG_B = (30, 34, 44)
COLOR_HUD_BG = (18, 20, 26)
COLOR_HUD_BORDER = (70, 200, 120)
COLOR_SNAKE_HEAD = (110, 231, 160)
COLOR_SNAKE_BODY_A = (60, 179, 113)
COLOR_SNAKE_BODY_B = (46, 150, 93)
COLOR_FOOD = (235, 87, 87)
COLOR_FOOD_HIGHLIGHT = (255, 150, 140)
COLOR_TEXT = (235, 235, 235)
COLOR_TEXT_DIM = (150, 155, 165)
COLOR_GAMEOVER_OVERLAY = (10, 10, 14)

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

    def __init__(self, render=False, max_steps_without_food=100, speed=DEFAULT_SPEED, label=None):
        self.render_enabled = render
        self.max_steps_without_food = max_steps_without_food
        self.speed = speed
        self.label = label
        self.episode = None
        self.total_episodes = None
        self.high_score = 0
        self._screen = None
        self._clock = None
        self._font = None
        self._font_small = None
        if self.render_enabled:
            self._init_render()
        self.reset()

    def _init_render(self):
        import pygame

        pygame.init()
        self._pygame = pygame
        window_size = (GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE + HUD_HEIGHT)
        self._screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Snake - Gaming Agent")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("segoeui", 22, bold=True)
        self._font_small = pygame.font.SysFont("segoeui", 16)

    def set_episode_info(self, episode=None, total_episodes=None, label=None):
        """Met à jour le bandeau HUD (numéro d'épisode, libellé de l'agent affiché)."""
        if episode is not None:
            self.episode = episode
        if total_episodes is not None:
            self.total_episodes = total_episodes
        if label is not None:
            self.label = label

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
            self.high_score = max(self.high_score, self.score)
            if self.render_enabled:
                self._draw(game_over=True)
                self._pygame.time.delay(500)
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
            self._clock.tick(self.speed)

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

    def get_state(self):
        """État discret sous forme de 11 booléens (dangers, direction, position nourriture).

        Une version à 14 booléens (danger anticipé à 2 cases) a été testée (voir NOTEBOOK.md,
        essai 5) : pas d'amélioration nette et confondue avec d'autres changements, donc pas
        retenue pour rester sur l'état le plus simple qui fonctionne.
        """
        head = self.head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # danger tout droit
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # danger à droite
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # danger à gauche
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
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

    def _draw_grid(self, pygame):
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                color = COLOR_BG_A if (gx + gy) % 2 == 0 else COLOR_BG_B
                pygame.draw.rect(
                    self._screen,
                    color,
                    (gx * BLOCK_SIZE, HUD_HEIGHT + gy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                )

    def _draw_snake(self, pygame):
        n = len(self.snake)
        for i, p in enumerate(self.snake):
            is_head = i == 0
            color = COLOR_SNAKE_HEAD if is_head else (COLOR_SNAKE_BODY_A if i % 2 else COLOR_SNAKE_BODY_B)
            rect = (p.x * BLOCK_SIZE + 1, HUD_HEIGHT + p.y * BLOCK_SIZE + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            pygame.draw.rect(self._screen, color, rect, border_radius=8 if is_head else 5)

            if is_head:
                self._draw_eyes(pygame, p)

    def _draw_eyes(self, pygame, head):
        cx = head.x * BLOCK_SIZE + BLOCK_SIZE // 2
        cy = HUD_HEIGHT + head.y * BLOCK_SIZE + BLOCK_SIZE // 2
        offset = BLOCK_SIZE // 4
        if self.direction in (Direction.LEFT, Direction.RIGHT):
            dx = offset if self.direction == Direction.RIGHT else -offset
            eyes = [(cx + dx, cy - offset), (cx + dx, cy + offset)]
        else:
            dy = offset if self.direction == Direction.DOWN else -offset
            eyes = [(cx - offset, cy + dy), (cx + offset, cy + dy)]
        for ex, ey in eyes:
            pygame.draw.circle(self._screen, (20, 30, 25), (ex, ey), 3)

    def _draw_food(self, pygame):
        cx = self.food.x * BLOCK_SIZE + BLOCK_SIZE // 2
        cy = HUD_HEIGHT + self.food.y * BLOCK_SIZE + BLOCK_SIZE // 2
        radius = BLOCK_SIZE // 2 - 2
        pygame.draw.circle(self._screen, COLOR_FOOD, (cx, cy), radius)
        pygame.draw.circle(self._screen, COLOR_FOOD_HIGHLIGHT, (cx - radius // 3, cy - radius // 3), max(2, radius // 3))

    def _draw_hud(self, pygame):
        pygame.draw.rect(self._screen, COLOR_HUD_BG, (0, 0, GRID_W * BLOCK_SIZE, HUD_HEIGHT))
        pygame.draw.line(self._screen, COLOR_HUD_BORDER, (0, HUD_HEIGHT - 1), (GRID_W * BLOCK_SIZE, HUD_HEIGHT - 1), 2)

        score_text = self._font.render(f"Score {self.score}", True, COLOR_TEXT)
        self._screen.blit(score_text, (14, 6))

        best_text = self._font_small.render(f"Meilleur: {self.high_score}", True, COLOR_TEXT_DIM)
        self._screen.blit(best_text, (14, 32))

        if self.label:
            label_surf = self._font_small.render(self.label, True, COLOR_TEXT_DIM)
            self._screen.blit(label_surf, (GRID_W * BLOCK_SIZE - label_surf.get_width() - 14, 8))

        if self.episode is not None:
            ep_str = f"Partie {self.episode}/{self.total_episodes}" if self.total_episodes else f"Partie {self.episode}"
            ep_surf = self._font_small.render(ep_str, True, COLOR_TEXT_DIM)
            self._screen.blit(ep_surf, (GRID_W * BLOCK_SIZE - ep_surf.get_width() - 14, 32))

    def _draw_game_over(self, pygame):
        overlay = pygame.Surface((GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE), pygame.SRCALPHA)
        overlay.fill((*COLOR_GAMEOVER_OVERLAY, 160))
        self._screen.blit(overlay, (0, HUD_HEIGHT))

        text = self._font.render(f"Partie terminée — score {self.score}", True, COLOR_TEXT)
        rect = text.get_rect(center=(GRID_W * BLOCK_SIZE // 2, HUD_HEIGHT + GRID_H * BLOCK_SIZE // 2))
        self._screen.blit(text, rect)

    def _draw(self, game_over=False):
        pygame = self._pygame
        self._draw_grid(pygame)
        self._draw_food(pygame)
        self._draw_snake(pygame)
        self._draw_hud(pygame)
        if game_over:
            self._draw_game_over(pygame)
        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            self._pygame.quit()
