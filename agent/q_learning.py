"""Agent Q-learning pour Snake avec récompense orientée vers la nourriture.

L’agent privilégie les actions sûres et les déplacements qui rapprochent la tête de la
nourriture, ce qui donne une meilleure stabilité que le Q-learning basique.
"""
import pickle
import random
from pathlib import Path

import numpy as np

N_ACTIONS = 3


class QLearningAgent:
    def __init__(self, lr=0.25, gamma=0.98, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.99):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _state_key(self, state):
        arr = np.asarray(state, dtype=float).reshape(-1)
        return tuple(float(x) for x in arr)

    def _ensure_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(N_ACTIONS, dtype=float)

    def _food_vector(self, state):
        # Les 4 derniers booléens sont toujours food_left/right/up/down, quel que soit le
        # nombre de bits de danger en tête du vecteur d'état.
        if len(state) < 11:
            return 0, 0
        dx = 0
        dy = 0
        if state[-4]:
            dx -= 1
        if state[-3]:
            dx += 1
        if state[-2]:
            dy -= 1
        if state[-1]:
            dy += 1
        return dx, dy

    def _distance_to_food(self, state):
        dx, dy = self._food_vector(state)
        return abs(dx) + abs(dy)

    def _food_bias(self, state):
        # Favorise les actions qui rapprochent la tête de la nourriture.
        # Les 4 booléens juste avant la position de la nourriture sont dir_l/r/u/d.
        dx, dy = self._food_vector(state)
        direction = np.argmax(np.asarray(state[-8:-4], dtype=float))
        action_scores = np.zeros(N_ACTIONS, dtype=float)

        for action in range(N_ACTIONS):
            if action == 0:
                target_dir = direction
            elif action == 1:
                target_dir = (direction + 1) % 4
            else:
                target_dir = (direction - 1) % 4

            if target_dir == 0 and dy < 0:
                action_scores[action] += 2.0
            if target_dir == 1 and dx > 0:
                action_scores[action] += 2.0
            if target_dir == 2 and dy > 0:
                action_scores[action] += 2.0
            if target_dir == 3 and dx < 0:
                action_scores[action] += 2.0

        return action_scores

    def _action_bias(self, state):
        return 1.5 * self._food_bias(state)

    def choose_action(self, state, greedy=False):
        key = self._state_key(state)
        self._ensure_state(key)

        danger = np.asarray(state[:3], dtype=float)
        safe_actions = [a for a in range(N_ACTIONS) if danger[a] == 0]

        if not greedy and random.random() < self.epsilon:
            if safe_actions:
                return random.choice(safe_actions)
            return random.randrange(N_ACTIONS)

        base_action = int(np.argmax(self.q_table[key] + self._action_bias(state)))
        if safe_actions:
            bias = self._action_bias(state)
            best_safe = max(safe_actions, key=lambda a: self.q_table[key][a] + bias[a])
            return best_safe
        return max(range(N_ACTIONS), key=lambda a: self.q_table[key][a]) if not greedy else base_action

    def update(self, state, action, reward, next_state, done):
        key = self._state_key(state)
        next_key = self._state_key(next_state)
        self._ensure_state(key)
        self._ensure_state(next_key)

        old_dist = self._distance_to_food(state)
        new_dist = self._distance_to_food(next_state)
        if old_dist > 0:
            reward += max(0.0, old_dist - new_dist) * 2.5

        if done:
            reward -= 15.0 if reward < 0 else 0.0

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = data["q_table"]
        return agent


class ImprovedQLearningAgent(QLearningAgent):
    pass
