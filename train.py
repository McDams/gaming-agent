"""Entraîne l'agent Q-learning sur Snake et sauvegarde la meilleure Q-table.

Usage:
    python train.py --episodes 1000 --run-name essai1
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.q_learning import ImprovedQLearningAgent
from game.snake_env import SnakeEnv


def _greedy_eval(agent, env, episodes=5):
    """Score moyen en politique greedy (sans exploration), pour juger la Q-table
    indépendamment du bruit de l'exploration epsilon-greedy en cours d'entraînement."""
    eval_scores = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, _, done, score = env.step(action)
        eval_scores.append(score)
    return float(np.mean(eval_scores))


def train(episodes=1000, run_name="run", seed=None, eval_every=100, eval_episodes=5):
    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)

    env = SnakeEnv(render=False)
    agent = ImprovedQLearningAgent()

    scores = []
    best_eval_avg = -1.0
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()
        scores.append(score)

        # Le score brut d'un épisode d'entraînement est bruité par l'exploration
        # epsilon-greedy : un coup de chance peut battre le record sans que la Q-table
        # soit vraiment meilleure. On juge donc et on sauvegarde sur une évaluation
        # greedy périodique, plus fiable (mêmes conditions que evaluate.py).
        if (ep + 1) % eval_every == 0:
            eval_avg = _greedy_eval(agent, env, episodes=eval_episodes)
            if eval_avg > best_eval_avg:
                best_eval_avg = eval_avg
                agent.save(run_dir / "best_agent.pkl")

        if (ep + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"Episode {ep + 1}/{episodes} - score: {score} - moyenne(50): {avg:.2f} - epsilon: {agent.epsilon:.3f}")

    agent.save(run_dir / "final_agent.pkl")

    with open(run_dir / "scores.json", "w") as f:
        json.dump(scores, f)

    _plot_curve(scores, run_dir / "learning_curve.png", run_name)

    print(f"\nMeilleure moyenne greedy (éval périodique sur {eval_episodes} parties): {best_eval_avg:.2f}")
    print(f"Score moyen (100 derniers épisodes d'entraînement): {np.mean(scores[-100:]):.2f}")
    print(f"Agents sauvegardés dans: {run_dir}")

    return scores


def _plot_curve(scores, out_path, run_name, window=50):
    scores = np.array(scores)
    moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, label="Score par épisode")
    plt.plot(range(window - 1, len(scores)), moving_avg, label=f"Moyenne mobile ({window})")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.title(f"Courbe de progression - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Courbe sauvegardée: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(episodes=args.episodes, run_name=args.run_name, seed=args.seed)
