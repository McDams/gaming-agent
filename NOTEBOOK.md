# Carnet d'essais

Format par entrée : date, changement testé, résultat chiffré, ce qu'on en conclut.
Inclure les tentatives ratées.

## Essai 0 — Référence aléatoire
- Date : 2026-08-20
- Agent aléatoire (`game/play_random.py`), 20 parties.
- Score moyen : 0.20 (max 1, min 0)
- Sert de référence pour tout le reste.

## Essai 1 — Q-learning, config par défaut, 300 épisodes (smoke test)
- Date : 2026-08-20
- lr=0.1, gamma=0.9, epsilon_decay=0.995, seed=42
- Juste un test rapide pour vérifier que le pipeline entraîne/sauvegarde/recharge
  correctement avant de lancer un run plus long.
- Score moyen (10 parties, greedy, agent rechargé) : 6.90 (max 20, min 0)
- Conclusion : le pipeline fonctionne, l'agent apprend déjà nettement mieux que le hasard
  en seulement 300 épisodes. On passe à un run plus long pour les résultats officiels.

## Essai 2 — Q-learning, config par défaut, 2000 épisodes (run officiel)
- Date : 2026-08-20
- lr=0.1, gamma=0.9, epsilon_decay=0.995, seed=42, run: `essai1`
- Moyenne mobile (50 derniers épisodes d'entraînement) : passe de ~0.2 à ~20 sur les 2000
  épisodes, se stabilise autour de 15-22 après ~800 épisodes (epsilon proche du minimum).
- Meilleur score en entraînement : 50
- Évaluation finale (best_agent.pkl rechargé depuis `evaluate.py`, 20 parties, greedy) :
  score moyen 19.00 (max 38, min 9) — comparable à l'agent aléatoire (0.20), soit ~95x
  mieux.
- Conclusion : la représentation d'état à 11 booléens (dangers + direction + position
  nourriture) suffit à un Q-learning tabulaire pour apprendre une politique largement
  meilleure que le hasard, sans réseau de neurones.

## Essai 3 — Reproductibilité, même config, seed différente
- Date : 2026-08-20
- Même hyperparamètres que l'essai 2, seed=7, run: `essai1_rerun`, 2000 épisodes.
- Moyenne mobile finale : 20.32-20.86 en fin d'entraînement (proche de l'essai 2).
- Score moyen sur 100 derniers épisodes d'entraînement : 19.18 (vs 19.56 pour l'essai 2)
- Meilleur score en entraînement : 46 (vs 50 pour l'essai 2)
- Conclusion : les deux courbes se ressemblent fortement (même ordre de grandeur, même
  forme générale de progression). Le Q-learning tabulaire sur cet espace d'états est donc
  raisonnablement stable d'un run à l'autre avec cette config — pas de signe
  d'instabilité ou de surapprentissage erratique.

## Essai 4 — Sweep 100 seeds x 3000 épisodes (recherche du meilleur agent)
- Date : 2026-08-21
- Objectif : entraîner un agent par seed (1 à 100), même config par défaut
  (lr=0.25, gamma=0.98, epsilon_decay=0.9995), 3000 épisodes chacun, pour voir si
  la seule variation de seed permet de dépasser le meilleur score connu (66, seed 13,
  trouvé lors d'un sweep préliminaire identique) et si possible atteindre 100+.
- Script : `sweep.py`, parallélisé (`ProcessPoolExecutor`). Pour chaque seed : entraînement
  complet + réévaluation greedy sur 20 parties (comme `evaluate.py`) pour avoir un score
  comparable d'un seed à l'autre (le score "best" vu pendant l'entraînement est bruité par
  l'exploration epsilon-greedy, epsilon ne descend qu'à ~0.22 après 3000 épisodes avec
  epsilon_decay=0.9995).
- Résultat sur les 100 seeds :
  - `train_best` (score max vu en entraînement) : moyenne 49.9, médiane 51, **max 66**
    (seed 13 — résultat identique au sweep préliminaire, logique car même code/seed).
    Seulement 1/100 seeds atteint 66, 7/100 atteignent ≥60, **aucun n'atteint 100**.
  - `eval_avg` (score moyen sur 20 parties, politique greedy, métrique la plus fiable) :
    moyenne 21.4, médiane 23.1, **meilleur agent : seed 91, eval_avg=35.55, eval_max=53**
    (train_best=61 pour ce seed — moins que le seed 13 en entraînement brut, mais plus
    régulier une fois greedy).
- Conclusion / échec instructif : **l'objectif de dépasser 66 n'est pas atteint** — le
  score plafonne exactement à 66 quelle que soit la seed testée parmi 1-100, et viser 100+
  n'est pas réaliste avec cette config (état à 11 booléens, Q-learning tabulaire, mêmes
  hyperparamètres). La seed seule ne suffit pas à améliorer significativement l'agent :
  ça montre que le plafond vient de la représentation d'état / des hyperparamètres, pas de
  la chance du tirage aléatoire. Pour aller plus loin il faudrait changer autre chose
  (state plus riche, plus d'épisodes avec une décroissance d'epsilon plus rapide, ou
  reward shaping) — cf. "Ce qu'on ferait avec plus de temps" dans le README.
- Agent retenu comme "meilleur agent" du projet : seed 91 (meilleur eval_avg, donc le plus
  fiable en greedy), copié dans `deliverable/best_agent.pkl` avec sa courbe
  (`deliverable/learning_curve.png`) et le résumé complet des 100 seeds
  (`deliverable/sweep_summary.csv`).
- Coût : 100 entraînements × 3000 épisodes = 300 000 épisodes, ~169 minutes en parallèle
  (3 workers, machine 4 cœurs physiques).

## Essai 5 — État enrichi à 14 booléens (danger anticipé à 2 cases) — RATÉ
- Date : 2026-08-21
- Changement : ajout de 3 booléens "danger à 2 cases" (tout droit/droite/gauche) en plus
  des 3 "danger à 1 case" existants, plus un biais d'action correspondant, en même temps
  qu'un passage d'`epsilon_decay` de 0.9995 à 0.999.
- Résultat (10 seeds x 3000 épisodes) : eval_avg moyen tombe de ~25.8 à ~11.0 — net recul.
- Diagnostic : deux changements testés en même temps (contraire à la règle "une amélioration
  à la fois"), donc impossible de savoir lequel est responsable. Un test isolé ultérieur
  (état 14 bits seul, epsilon_decay inchangé) montre un résultat toujours mitigé/pas
  meilleur que l'état à 11 bits, selon la config d'epsilon testée.
- Conclusion : **annulé**, retour à l'état à 11 booléens d'origine. Le gain espéré
  (anticiper les impasses) ne s'est pas confirmé, sans doute parce que la table Q doit déjà
  apprendre ~8x plus de combinaisons d'états pour le même budget d'épisodes.

## Essai 6 — Bug d'exploration epsilon-greedy corrigé + recalibrage — RÉUSSI
- Date : 2026-08-21
- Découverte (en même temps qu'une correction apportée en parallèle sur le code) : dans
  l'agent original, la branche d'exploration aléatoire d'epsilon-greedy calculait bien une
  action aléatoire, mais elle était systématiquement écrasée par le filtre de sécurité qui
  repassait en glouton (Q + biais nourriture) dès qu'une action sûre existait — c'est-à-dire
  presque tout le temps. Résultat : `epsilon` n'avait quasiment aucun effet, l'agent jouait
  déjà en mode quasi-glouton dès l'épisode 1, guidé par l'heuristique nourriture. C'est ce
  qui explique en partie les bons scores obtenus dans les essais précédents malgré un
  epsilon nominal élevé.
- Correction : l'exploration aléatoire choisit maintenant vraiment une action au hasard
  (parmi les actions sûres si possible), comme un epsilon-greedy est censé fonctionner.
  Comportement plus correct, mais qui casse les scores tel quel : avec `epsilon_decay`
  toujours à 0.9995/0.999 (jamais calibré pour une vraie exploration), l'agent erre au
  hasard une bonne partie des 3000 épisodes au lieu de suivre l'heuristique.
- Recalibrage : `epsilon_decay` passé à 0.99 (epsilon atteint son minimum ~0.02 vers
  l'épisode 390, donc l'agent a largement le temps de converger sur 3000 épisodes tout en
  ayant vraiment exploré au début).
- Bonus découvert en testant : la sauvegarde de `best_agent.pkl` se faisait auparavant sur
  le score brut d'un épisode d'entraînement, bruité par l'exploration — un cas observé :
  train_best=69 en entraînement mais eval_avg=1.1 une fois rechargé en greedy (coup de
  chance non représentatif). Corrigé dans `train.py` : la sauvegarde se fait maintenant sur
  une évaluation greedy périodique (5 parties toutes les 100 épisodes), cohérente avec la
  façon dont `evaluate.py` juge l'agent.
- Résultat (10 seeds x 3000 épisodes, mêmes seeds que l'essai 4) :

  | | eval_avg moyen | eval_max (meilleur score) |
  |---|---|---|
  | Avant (essai 4, bug d'exploration) | 25.8 | 50 |
  | Après (essai 6) | 29.5 | 68 |

  Plus de scores catastrophiques après rechargement (l'écart train_best/eval_avg observé
  avant la correction du bonus de sauvegarde a disparu).
- Sweep complet 100 seeds x 3000 épisodes relancé avec cette config pour trouver l'agent
  définitif (voir résultats mis à jour dans le README et `deliverable/`).
