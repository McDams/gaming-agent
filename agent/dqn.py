"""Agent DQN (Deep Q-Network) pour Snake — même état à 11 booléens et même récompense
(façonnée par la distance à la nourriture) que l'agent Q-learning tabulaire, pour une
comparaison équitable. Contrairement à l'agent tabulaire, il n'a pas d'heuristique de
biais nourriture câblée en dur dans le choix d'action : le réseau doit apprendre à
chercher la nourriture uniquement à partir de la récompense.
"""
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Réseau minuscule : le multithreading BLAS coûte plus qu'il ne rapporte, et on veut
# pouvoir paralléliser plusieurs seeds par process (comme pour l'agent tabulaire).
torch.set_num_threads(1)

STATE_DIM = 11
N_ACTIONS = 3


class QNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        lr=2.5e-4,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        buffer_size=50_000,
        batch_size=32,
        tau=0.01,
        min_buffer=1000,
        train_every=4,
        grad_clip=10.0,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.min_buffer = min_buffer
        self.train_every = train_every
        self.grad_clip = grad_clip

        self.policy_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0

    def _distance_to_food(self, state):
        dx = -1 if state[-4] else (1 if state[-3] else 0)
        dy = -1 if state[-2] else (1 if state[-1] else 0)
        return abs(dx) + abs(dy)

    def choose_action(self, state, greedy=False):
        danger = state[:3]
        safe_actions = [a for a in range(N_ACTIONS) if danger[a] == 0]

        if not greedy and random.random() < self.epsilon:
            if safe_actions:
                return random.choice(safe_actions)
            return random.randrange(N_ACTIONS)

        with torch.no_grad():
            q = self.policy_net(torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0))
            q = q.squeeze(0).numpy()
        if safe_actions:
            return max(safe_actions, key=lambda a: q[a])
        return int(np.argmax(q))

    def update(self, state, action, reward, next_state, done):
        old_dist = self._distance_to_food(state)
        new_dist = self._distance_to_food(next_state)
        if old_dist > 0:
            reward += max(0.0, old_dist - new_dist) * 2.5
        if done:
            reward -= 15.0 if reward < 0 else 0.0

        self.buffer.append((np.asarray(state, dtype=np.float32), action, reward, np.asarray(next_state, dtype=np.float32), float(done)))
        self.steps += 1

        if len(self.buffer) < self.min_buffer or self.steps % self.train_every != 0:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN : le réseau policy choisit l'action, le réseau cible l'évalue.
            # Réduit le biais de surestimation qui rendait l'entraînement instable.
            next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Mise à jour douce (Polyak) du réseau cible plutôt qu'une synchronisation brutale :
        # une cible qui bouge trop vite était une des causes probables de l'instabilité.
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * policy_param)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu")
        agent = cls(epsilon=data.get("epsilon", 0.0))
        agent.policy_net.load_state_dict(data["policy_state_dict"])
        agent.target_net.load_state_dict(data["policy_state_dict"])
        agent.target_net.eval()
        return agent
