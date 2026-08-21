"""Recharge un agent sauvegardé (script indépendant de train.py) et le fait rejouer.

Usage:
    python evaluate.py --model runs/run/best_agent.pkl --episodes 20 --render
"""
import argparse

import numpy as np

from agent.q_learning import QLearningAgent
from game.snake_env import SnakeEnv


def evaluate(model_path, episodes=20, render=False, speed=None):
    agent = QLearningAgent.load(model_path)
    env = SnakeEnv(render=render, speed=speed or 12, label="Agent entraîné (Q-learning)")

    scores = []
    for ep in range(episodes):
        state = env.reset()
        env.set_episode_info(episode=ep + 1, total_episodes=episodes)
        done = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, _, done, score = env.step(action)
        scores.append(score)
        print(f"Episode {ep + 1}/{episodes} - score: {score}")

    env.close()
    print(f"\nScore moyen agent entraîné sur {episodes} parties : {np.mean(scores):.2f}")
    print(f"Max: {max(scores)} | Min: {min(scores)}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--speed", type=int, default=None, help="FPS du rendu (défaut: 12)")
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes, render=args.render, speed=args.speed)
