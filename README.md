# Gaming Agent — Snake + Q-Learning

## Équipe
- Nom de team : _à compléter_
- Membres : _à compléter_

## Le jeu
Snake, implémenté à la main avec PyGame (grille 20x15 cases). Choisi car l'environnement
tourne et donne un score en quelques minutes de mise en place, sans dépendance lourde
(pas de GPU nécessaire), et permet un espace d'états discrétisable simplement pour du
Q-learning tabulaire.

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

Hyperparamètres par défaut (`agent/q_learning.py`) :
- learning rate = 0.1
- gamma (discount) = 0.9
- epsilon initial = 1.0, décroissance ×0.995 par épisode, minimum 0.01

## Résultats

| Agent | Score moyen | Max | Min | Nb parties |
|---|---|---|---|---|
| Aléatoire (référence) | 0.20 | 1 | 0 | 20 |
| Q-learning entraîné (2000 épisodes) | 19.00 | 38 | 9 | 20 |

L'agent entraîné fait ~95x mieux que le hasard sur le même nombre de parties.
Entraînement relancé une seconde fois (seed différente) : moyenne mobile finale de 19.18
contre 19.56 pour le premier run, meilleur score 46 contre 50 — comportement stable et
reproductible d'un run à l'autre.

Courbe de progression : voir `runs/<nom_run>/learning_curve.png` (générée localement,
non versionnée — voir [NOTEBOOK.md](NOTEBOOK.md) pour l'historique complet des essais).

## Comment lancer

```bash
pip install -r requirements.txt

# Agent aléatoire jouable (référence)
python game/play_random.py --episodes 20 --render

# Entraînement
python train.py --episodes 1000 --run-name essai1

# Recharger le meilleur agent et le faire rejouer (script indépendant)
python evaluate.py --model runs/essai1/best_agent.pkl --episodes 20 --render
```

## Carnet d'essais
Voir [NOTEBOOK.md](NOTEBOOK.md) pour l'historique des tentatives, y compris les échecs.

## Vidéo de présentation
Lien : _à compléter_

## Ce qu'on ferait avec plus de temps
- Passer à une représentation d'état plus riche (distance normalisée à la nourriture,
  vision sur plusieurs cases dans chaque direction) ou à un DQN (réseau de neurones) pour
  dépasser les limites d'une table Q sur un état binaire à 11 dimensions.
- Complexifier la fonction de récompense (pénaliser les trajectoires qui s'éloignent
  durablement de la nourriture, encourager la survie) une fois la version simple
  validée, comme le suggère la consigne du projet.
- Recherche d'hyperparamètres plus systématique (learning rate, gamma, vitesse de
  décroissance d'epsilon) plutôt que la config par défaut réutilisée pour tous les essais
  — voir le sweep multi-seeds dans le carnet d'essais, qui va dans ce sens pour la seed
  mais pas encore pour les autres hyperparamètres.
- Automatiser la capture vidéo de l'agent en action plutôt qu'une capture manuelle.
