# Gaming Agent — Snake + Q-Learning

## Équipe
- Nom de team : _à compléter_
- Membres : _à compléter_

## Le jeu
Snake, implémenté à la main avec PyGame (grille 20x15 cases). Choisi car l'environnement
tourne et donne un score en quelques minutes de mise en place, sans dépendance lourde
(pas de GPU nécessaire), et permet un espace d'états discrétisable simplement pour du
Q-learning tabulaire.

Rendu graphique : bandeau HUD (score, meilleur score, numéro de partie, libellé de l'agent
affiché), grille en damier, tête du serpent distincte du corps (avec des yeux qui indiquent
la direction), nourriture ronde, écran de fin de partie. Vitesse d'affichage réglable
(`--speed`, cf. plus bas) pour rester lisible en vidéo.

## Ce que l'agent observe, fait, et ce qui le récompense

**Observation (état, 11 booléens)** :
- 3 dangers immédiats (tout droit / à droite / à gauche du sens de déplacement actuel)
- direction actuelle du serpent (haut/bas/gauche/droite)
- position relative de la nourriture (gauche/droite/haut/bas par rapport à la tête)

Cela donne un espace d'états discret (jusqu'à 2^11 = 2048 combinaisons), assez petit pour
une table Q classique.

**Actions (3, relatives à la direction courante)** : tout droit, tourner à droite, tourner à gauche.

**Récompense** :
- +10 quand le serpent mange la nourriture
- -10 quand la partie se termine (collision avec un mur ou avec lui-même, ou trop de pas
  sans manger)
- 0 sinon

Fonction volontairement simple, conforme à la consigne "simple d'abord, complexifiée si le
temps le permet".

## Méthode d'apprentissage : Q-learning tabulaire
Choisi parce que l'état est discrétisé en un nombre raisonnable de combinaisons (≤2048),
ce qui rend une table Q (dictionnaire état→valeurs d'actions) suffisante, sans avoir besoin
d'un réseau de neurones. C'est aussi la méthode la plus simple à faire tourner et à
comprendre entièrement en 2 jours, sans GPU.

Deux alternatives ont été testées pour essayer de l'améliorer : Double Q-learning (plus
régulier mais pas meilleur sur ce budget d'épisodes) et un DQN (réseau de neurones, plafonne
nettement en dessous même avec 3x plus d'épisodes). Aucune des deux n'a été retenue — détail
et raisons dans [NOTEBOOK.md](NOTEBOOK.md), essais 7 et 8.

Hyperparamètres par défaut (`agent/q_learning.py`) :
- learning rate = 0.25
- gamma (discount) = 0.98
- epsilon initial = 1.0, décroissance ×0.99 par épisode, minimum 0.02 (retendu à l'essai 6
  après correction d'un bug d'exploration — voir [NOTEBOOK.md](NOTEBOOK.md))

## Résultats

| Agent | Score moyen | Max | Min | Nb parties |
|---|---|---|---|---|
| Aléatoire (référence) | 0.20 | 1 | 0 | 20 |
| Q-learning entraîné (2000 épisodes, 1 run) | 19.00 | 38 | 9 | 20 |
| **Meilleur agent retenu** (sweep 100 seeds × 3000 épisodes, seed 36) | **42.15** | 61 | 14 | 20 |

Le meilleur agent retenu fait ~211x mieux que le hasard sur le même nombre de parties.
Reproductibilité vérifiée à plusieurs niveaux : un run relancé avec une seed différente
(essai 3) donne des courbes très proches ; un sweep sur 100 seeds indépendantes (essai 4,
avant un correctif décrit ci-dessous) montre une distribution stable (score max en
entraînement : moyenne 49.9, écart-type 8.1). Un bug d'exploration epsilon-greedy a ensuite
été corrigé et les hyperparamètres retendus en conséquence (essai 6) : rejoué sur les 100
mêmes seeds, le score moyen monte à 56.3 (écart-type 4.8, donc plus régulier) et le meilleur
score passe de 66 à 70 — voir [NOTEBOOK.md](NOTEBOOK.md) pour le détail complet, y compris
les tentatives ratées (état enrichi qui n'a pas aidé, sauvegarde sur un score d'entraînement
bruité qui donnait parfois un agent malchanceux une fois rechargé).

Après ça, deux pistes pour aller plus loin que le Q-learning simple ont été testées, sur le
même budget de 3000 épisodes (sauf mention contraire) et la même méthode d'évaluation :

| Variante | eval_avg moyen | eval_avg max | Retenue ? |
|---|---|---|---|
| **Q-learning simple (retenu)** | **29.3** | **42.15** | ✅ |
| Double Q-learning | 27.7 (écart-type ~3x plus faible) | 30.65 | ❌ (essai 7) |
| DQN, réseau de neurones (10 000 épisodes) | 27.0 | 45 | ❌ (essai 8) |

Aucune des deux ne bat le Q-learning simple sur ce budget d'épisodes — Double Q-learning est
plus régulier mais plus lent à converger (deux tables à remplir au lieu d'une), le DQN
plafonne et oscille sans jamais vraiment converger. Détail complet, y compris le
diagnostic de chaque échec, dans [NOTEBOOK.md](NOTEBOOK.md) (essais 7 et 8).

Courbe de progression du meilleur agent : [deliverable/learning_curve.png](deliverable/learning_curve.png).
Résumé complet des 100 seeds testés : [deliverable/sweep_summary.csv](deliverable/sweep_summary.csv).
Historique complet des essais (y compris ratés) : [NOTEBOOK.md](NOTEBOOK.md).

## Comment lancer

```bash
pip install -r requirements.txt
```

### Voir jouer le meilleur agent entraîné (dernier modèle retenu)

```bash
python evaluate.py --model deliverable/best_agent.pkl --episodes 10 --render
```

`deliverable/best_agent.pkl` est le meilleur agent trouvé à ce jour (seed 36, sweep sur 100
seeds × 3000 épisodes — voir "Résultats" ci-dessus et [NOTEBOOK.md](NOTEBOOK.md)). Il se
recharge directement, sans réentraînement. `--episodes` contrôle le nombre de parties
jouées, `--speed N` la vitesse d'affichage en images/seconde (défaut 12, pensé pour rester
regardable ; monter à 30-40 pour aller plus vite, descendre à 6-8 pour un rendu plus posé
en vidéo).

### Autres commandes

```bash
# Agent aléatoire jouable (référence)
python game/play_random.py --episodes 20 --render

# Entraînement (sauvegarde périodiquement le meilleur agent, jugé en greedy)
python train.py --episodes 1000 --run-name essai1
python evaluate.py --model runs/essai1/best_agent.pkl --episodes 20 --render

# Relancer un sweep multi-seeds pour rechercher un meilleur agent
python sweep.py --episodes 3000 --seed-start 1 --seed-end 100 --workers 4
```

## Carnet d'essais
Voir [NOTEBOOK.md](NOTEBOOK.md) pour l'historique des tentatives, y compris les échecs.

## Vidéo de présentation
Lien : _à compléter_

## Ce qu'on ferait avec plus de temps
- On a testé un état enrichi à 14 booléens (danger anticipé à 2 cases) : pas concluant
  (essai 5), probablement parce que l'espace d'états devient ~8x plus grand pour le même
  budget d'épisodes. À retester isolément avec plus d'épisodes.
- On a aussi testé Double Q-learning (plus régulier, pas meilleur — essai 7) et un DQN
  (réseau de neurones, plafonne en dessous du tabulaire même à 10 000 épisodes — essai 8).
  Pour le DQN, les pistes non essayées faute de temps : normaliser/clipper la récompense,
  replay prioritisé, ou lui donner un état plus riche (justement ce qui n'a pas marché en
  tabulaire à l'essai 5) puisqu'un réseau généralise mieux qu'une table sur un état large —
  mais ça n'a pas pu être testé avant la deadline.
- Complexifier la fonction de récompense (pénaliser les trajectoires qui s'éloignent
  durablement de la nourriture, encourager la survie) une fois la version simple
  validée, comme le suggère la consigne du projet.
- Recherche d'hyperparamètres plus systématique (learning rate, gamma) : on a retendu
  `epsilon_decay` après avoir corrigé un bug d'exploration (essai 6, gain net), et exploré
  la seed sur 100 valeurs (essais 4 et 6), mais lr et gamma sont restés à leur valeur par
  défaut sur tous les essais.
- Automatiser la capture vidéo de l'agent en action plutôt qu'une capture manuelle.
