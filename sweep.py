"""Lance plusieurs entraînements en parallèle (seeds différentes) et garde le meilleur agent.

Chaque seed entraîne un agent indépendant sur `--episodes` épisodes, puis son
`best_agent.pkl` est réévalué en greedy (comme evaluate.py) pour avoir un score comparable
d'un seed à l'autre. Le meilleur agent du sweep (selon le score greedy) est copié à part.

Usage:
    python sweep.py --episodes 3000 --seed-start 1 --seed-end 100 --workers 6
"""
import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluate import evaluate
from train import train


def run_one(seed, episodes, sweep_name, eval_episodes):
    run_name = f"{sweep_name}/seed_{seed:03d}"
    t0 = time.time()
    scores = train(episodes=episodes, run_name=run_name, seed=seed)

    model_path = Path("runs") / run_name / "best_agent.pkl"
    eval_scores = evaluate(str(model_path), episodes=eval_episodes, render=False)

    return {
        "seed": seed,
        "train_best": int(max(scores)),
        "train_avg_last100": round(float(np.mean(scores[-100:])), 2),
        "eval_avg": round(float(np.mean(eval_scores)), 2),
        "eval_max": int(max(eval_scores)),
        "eval_min": int(min(eval_scores)),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--sweep-name", type=str, default="sweep")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_end + 1))
    sweep_dir = Path("runs") / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, s, args.episodes, args.sweep_name, args.eval_episodes): s
            for s in seeds
        }
        done_count = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            done_count += 1
            elapsed_total = time.time() - t_start
            try:
                res = fut.result()
                results.append(res)
                print(
                    f"[{done_count}/{len(seeds)}] seed={seed} "
                    f"train_best={res['train_best']} eval_avg={res['eval_avg']} "
                    f"eval_max={res['eval_max']} - elapsed_total={elapsed_total:.0f}s",
                    flush=True,
                )
            except Exception as e:
                results.append({"seed": seed, "error": str(e)})
                print(f"[{done_count}/{len(seeds)}] seed={seed} ERREUR: {e}", flush=True)

    results.sort(key=lambda r: r["seed"])

    with open(sweep_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if "error" not in r]
    with open(sweep_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "train_best",
                "train_avg_last100",
                "eval_avg",
                "eval_max",
                "eval_min",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for r in valid:
            writer.writerow(r)

    print("\n=== RESUME DU SWEEP ===")
    print(f"{len(valid)}/{len(seeds)} runs reussis")

    if valid:
        best = max(valid, key=lambda r: (r["eval_avg"], r["eval_max"]))
        best_model_src = Path("runs") / args.sweep_name / f"seed_{best['seed']:03d}" / "best_agent.pkl"
        best_model_dst = sweep_dir / "overall_best_agent.pkl"
        shutil.copy(best_model_src, best_model_dst)

        train_bests = [r["train_best"] for r in valid]
        eval_avgs = [r["eval_avg"] for r in valid]
        print(f"Meilleur seed (selon eval greedy): {best['seed']}")
        print(f"  eval_avg={best['eval_avg']} eval_max={best['eval_max']} train_best={best['train_best']}")
        print(f"Score train_best max tous seeds confondus: {max(train_bests)}")
        print(f"Score eval_avg max tous seeds confondus: {max(eval_avgs)}")
        print(f"Agent copie vers: {best_model_dst}")

    print(f"Resume: {sweep_dir / 'summary.csv'}")
    print(f"Temps total: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
