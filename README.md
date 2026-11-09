# MyTorch : analyseur de positions d'échecs

Un petit projet de réseau de neurones en NumPy, avec un analyseur en ligne de
commande pour des positions d'échecs écrites en notation FEN.

Le dépôt ne doit contenir qu'**un seul** fichier dont le nom commence par
`my_torch_network` (`my_torch_network_full.nn`) : la mouli le cherche avec
`find . -type f -name "my_torch_network*"` et ne doit trouver qu'un modèle.

## Prérequis

- Python 3.10 ou plus récent
- Les paquets de `requirements.txt` (`numpy` et `matplotlib`)

Installer les dépendances depuis la racine du dépôt :

```sh
# Créer un environnement virtuel si besoin
python -m venv .venv

# L'activer (Linux/macOS)
source .venv/bin/activate

# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Démarrage rapide

```sh
make                                                       # crée l'exécutable my_torch_analyzer
python3 scripts/build_dataset.py 50000 data/dataset.npz   # encode et équilibre assets/
python3 scripts/train.py --limit 200000 --hidden 128 64 --dropout 0.3 --batch-size 32 --l2 0.0001 --epochs 25 --patience 5 --out my_torch_network_full.nn
./my_torch_analyzer --predict my_torch_network_full.nn positions.txt
```

Le réseau pré-entraîné `my_torch_network_full.nn` est dans le dépôt.

## Lancer l'analyseur

Le point d'entrée en ligne de commande est `my_torch_analyzer` (créé par `make`
à partir de `my_torch_analyzer.py` ; `make re`, `make clean` et `make fclean`
existent) :

```sh
./my_torch_analyzer --help
./my_torch_analyzer --predict my_torch_network_full.nn positions.txt
./my_torch_analyzer --train my_torch_network_full.nn positions-etiquetees.txt
./my_torch_analyzer --train --save entraine.nn my_torch_network_full.nn positions-etiquetees.txt
```

Sans `--save`, l'entraînement met à jour le modèle de `LOADFILE`. Avec `--save`,
le modèle mis à jour est écrit dans `SAVEFILE` et `LOADFILE` n'est pas modifié.
Les erreurs sont écrites sur la sortie d'erreur et le code de retour est 84
(0 sinon).

Chaque ligne non vide d'un fichier d'échecs commence par une position FEN à six
champs. Les lignes d'entraînement doivent finir par une étiquette ; en
prédiction, l'étiquette est facultative :

```text
4k3/8/8/8/8/8/8/4K3 w - - 0 1 Nothing
```

En mode prédiction, une ligne est affichée par échiquier, dans le même ordre,
parmi `Nothing`, `Check White`, `Check Black`, `Checkmate White`,
`Checkmate Black`. La couleur est celle du camp qui donne l'échec. Un réseau
n'accepte que les étiquettes de ses cinq classes : entraîner avec `Check`,
`Checkmate` ou `Stalemate` est refusé avec une erreur (`assets/` ne contient
aucune position de pat).

Générer des réseaux vierges à partir d'un fichier de configuration :

```sh
./my_torch_generator configs/basic_network.conf 3   # basic_network_1.nn ... _3.nn
```

## Organisation du dépôt

| chemin | contenu |
|---|---|
| `source/nn/` | module de réseau de neurones : couches, activations, pertes, optimiseurs, `Network`, sauvegarde/chargement, générateur, recherche aléatoire |
| `source/analyzer/` | parseur FEN, encodage FEN, modes de la CLI, liaison entre la CLI et le réseau (`pipeline.py`) |
| `scripts/` | construction du dataset, entraînement, benchmarks (nécessaires pour reconstruire le réseau) |
| `bonus/` | recherche aléatoire d'hyperparamètres |
| `configs/` | exemple de configuration pour le générateur |
| `logs/` | un JSON par entraînement (configuration, graine, courbes, précisions, matrice de confusion) et les résultats des benchmarks |
| `docs/` | [chosen_config.md](docs/chosen_config.md), [training.md](docs/training.md), [benchmarks.md](docs/benchmarks.md), [benchmarks_tables.md](docs/benchmarks_tables.md), [nn.md](docs/nn.md), [analyser.md](docs/analyser.md), [test.md](docs/test.md) |

## Le réseau de neurones

Réseau : `971 -> 128 -> 64 -> 5`, ReLU, initialisation He sur les couches
cachées, Xavier sur la couche de sortie, softmax, perte d'entropie croisée,
Adam (0,001), mini-batchs de 32, dropout 0,3, L2 0,0001, arrêt anticipé. Chaque
valeur est justifiée, avec les images des benchmarks, dans
[docs/chosen_config.md](docs/chosen_config.md). NumPy pur, aucune bibliothèque
d'apprentissage automatique. Le fichier `.nn` est du JSON
(`"format": "mytorch-chess-v1"`, liste de couches avec poids et biais).

L'entrée compte 971 caractéristiques : one-hot de la pièce sur chaque case
(768), camp qui joue, droits de roque, case de prise en passant, cases
attaquées par chaque camp (128) et, pour chaque roi : en échec, nombre de
pièces qui l'attaquent, cases de fuite libres. Elles sont calculées à partir de
l'échiquier seul, jamais de l'étiquette. Détails dans
[docs/training.md](docs/training.md).

### Résultats

Réseau pré-entraîné, 200 000 positions équilibrées (160 000 entraînement /
20 000 validation / 20 000 test) :

| entraînement | validation | test | 25 000 positions mises de côté |
|---|---|---|---|
| 96,5 % | 96,3 % | 96,4 % | 96,3 % |

`Nothing` et la couleur du camp qui donne l'échec ne sont jamais confondus.
Toutes les erreurs sont entre un échec et un échec et mat de la même couleur
(le réseau estime si le roi peut fuir, mais ne simule pas le blocage ni la
capture de la pièce qui donne l'échec). Les précisions d'entraînement et de
validation sont presque égales : le réseau ne surapprend pas. Le réseau
précédent (batch 128, sans régularisation, 150 000 positions) obtenait 95,8 %
sur les mêmes positions mises de côté ; il est conservé dans
`bonus/bonus_network_batch128.nn`. Courbes, matrice de confusion et détails :
[docs/chosen_config.md](docs/chosen_config.md).

Test supplémentaire sur 30 000 positions de `assets/` jamais vues par le réseau
(6 000 par classe) : 96,4 % au total ; 98,9 % avec 10 pièces, 96,5 % avec 20
pièces, 93,3 % avec beaucoup de pièces. Le réseau annonce parfois un mat alors
que c'est seulement un échec, presque jamais l'inverse.

## Choix de conception

Chaque choix, sa raison et les preuves. Les résultats mesurés viennent de
`scripts/benchmark.py` (une seule variable change à la fois, 24 000 positions,
8 époques, graine 0). Tableaux complets :
[docs/benchmarks_tables.md](docs/benchmarks_tables.md).

| choix | retenu | essayé / rejeté | pourquoi |
|---|---|---|---|
| activation cachée | ReLU | leaky ReLU, tanh, sigmoïde | ReLU, leaky ReLU et tanh sont équivalentes (précision test 0,920 / 0,921 / 0,920). La sigmoïde est moins bonne (0,875) : sa dérivée vaut au plus 0,25, le gradient diminue de couche en couche. ReLU est la moins coûteuse et ne sature pas. |
| activation de sortie | softmax | sigmoïde, aucune | cinq classes exclusives demandent des probabilités de somme 1 ; softmax avec l'entropie croisée donne le gradient simple `(p - y) / N`. |
| perte | entropie croisée | MSE | la MSE a seulement servi à valider la bibliothèque sur le XOR ; l'entropie croisée est la perte faite pour la classification. |
| initialisation | He (cachées), Xavier (sortie) | Xavier, constante 0,01 | He 0,920 contre Xavier 0,917 avec ReLU (dans le bruit, mais cohérent avec la théorie : He compense la moitié du signal que ReLU supprime). Une échelle constante de 0,01 bloquait le réseau XOR à une perte de 0,25 : le signal s'éteint à travers les couches. |
| optimiseur | Adam (0,001) | SGD, momentum | SGD à 0,05 est le plus lent (0,881). Momentum 0,931 et Adam 0,920 sont proches ; Adam demande bien moins de réglage du learning rate. |
| learning rate | 0,001 | 0,0003, 0,01 | 0,0003 est trop lent (0,875) ; 0,01 est instable (perte d'entraînement plus basse mais perte de validation plus haute). |
| taille de batch | 32 | 128, 512 | à nombre d'époques fixé, un petit batch fait plus de mises à jour : 32 donne 0,924, 128 donne 0,920, 512 donne 0,860, pour un coût par époque à peu près identique. La recherche aléatoire va dans le même sens. |
| architecture | 128 -> 64 | 32, 64, 128, 64 -> 32 | la précision augmente avec la capacité puis sature (0,892, 0,907, 0,908, 0,920, 0,916). Le plus grand réseau surapprend sur 24 000 positions (entraînement 0,988, validation 0,920). 128 -> 64 est retenu car le jeu d'entraînement final est bien plus grand (160 000 positions) et le réseau est régularisé. |
| surapprentissage | dropout 0,3 + L2 1e-4 + arrêt anticipé (patience 5) + grand dataset | chacun seul, dropout 0,1 | sur 6 000 positions, le réseau simple atteint 1,000 en entraînement / 0,843 en validation et L2 l'améliore (0,863). Sur l'architecture retenue, la combinaison batch 32 + L2 + dropout 0,3 a la meilleure perte de validation (0,172 contre 0,207 pour le réglage précédent). L'arrêt anticipé restaure les meilleurs poids. |
| caractéristiques d'entrée | 971 avec cartes d'attaque | 837 brutes (pièces, camp, roque, prise en passant) | l'échec est une propriété géométrique qu'un petit réseau apprend mal à partir des cases brutes ; on lui donne donc des caractéristiques d'attaque et de roi. C'est de l'ingénierie de caractéristiques, le réseau apprend toujours l'étiquetage. L'encodage brut seul n'a pas été comparé. |

L2 et Adam : le réseau final applique L2 (1e-4) en ajoutant `l2 * w` au gradient
avant le pas d'Adam. Avec Adam, cette pénalité est divisée par l'échelle
adaptative de chaque poids : ce n'est donc pas une décroissance des poids
uniforme. `AdamW` (décroissance des poids découplée, `source/nn/optimizers.py`,
`--optimizer adamw --weight-decay`) est implémenté et testé unitairement, mais
le réseau final n'a pas été réentraîné avec ; son effet n'est donc pas mesuré.
Avec un `l2` aussi petit et sans surapprentissage (entraînement 96,5 % /
validation 96,3 %), la différence devrait être faible.

### Recherche aléatoire (bonus)

`bonus/hyperparameter_search.py` tire ensemble le learning rate, l'architecture,
la taille de batch, l'activation, le dropout et L2 (10 essais, 8 époques,
24 000 positions). Résultats dans `logs/hyperparameter_search.json` :

| rang | perte val. | précision val. | lr | cachées | batch | activation | dropout | L2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0,167 | 0,938 | 0,0014 | 32 | 32 | relu | 0,3 | 1e-4 |
| 2 | 0,177 | 0,934 | 0,0048 | 32 | 64 | leaky ReLU | 0,1 | 1e-4 |
| 3 | 0,184 | 0,929 | 0,0022 | 32 | 32 | leaky ReLU | 0,1 | 1e-3 |
| 5 | 0,212 | 0,918 | 0,0019 | 128 -> 64 | 256 | tanh | 0,3 | 0 |
| 10 | 0,351 | 0,859 | 0,0007 | 64 | 256 | relu | 0,1 | 1e-3 |

Lecture : les meilleurs essais utilisent tous un petit batch (32 ou 64), et les
pires un grand batch ou un learning rate bas. Avec un budget fixe de 8 époques,
la taille de batch domine car elle fixe le nombre de mises à jour ; le petit
réseau `[32]` n'est pas meilleur en soi, il s'entraîne simplement vite. Cela
rejoint les balayages un paramètre à la fois (batch 32 meilleur, 512 pire,
lr 0,0003 trop lent).

## Ce qu'on a essayé et qui n'a pas marché

- Échelle de poids constante 0,01 : XOR bloqué à une perte de 0,25 (signal qui s'éteint). Corrigé par Xavier / He.
- Sigmoïde dans les couches cachées : nettement moins bonne que ReLU / tanh.
- SGD simple : lent, demande un grand learning rate et beaucoup d'époques.
- Grands batchs (512) et learning rates bas (0,0003) : pas assez de mises à jour dans le budget d'époques.
- Dropout 0,3 seul sur un petit dataset : pas de gain réel ; il n'a aidé qu'avec L2 et un petit batch sur le plus grand réseau.
- Le gain de l'entraînement final sur le précédent est faible sur les données mises de côté (95,8 % à 96,3 %), bien moins que ce que suggérait la comparaison sur 24 000 positions.

## Limites et réserves

- Chaque variante des benchmarks n'a été lancée qu'une fois (graine 0). Les écarts
  inférieurs à environ 0,5 point sont du bruit. Pas de lancements avec plusieurs
  graines.
- Les balayages font varier une variable à la fois : les interactions ne sont pas
  mesurées ; la recherche aléatoire en couvre une partie, mais avec seulement 10
  essais.
- La configuration finale vient d'une comparaison de six combinaisons sur 24 000
  positions avec une seule graine, dont les courbes oscillent : une partie de
  l'avantage de la configuration retenue peut être du bruit.
- La perte de validation de l'entraînement final saute à l'époque 11 avant de
  redescendre (l'arrêt anticipé a rendu les poids de l'époque 7).
- Le jeu de test vient du même générateur que le jeu d'entraînement. Les données
  d'entraînement sont équilibrées ; les vraies parties contiennent surtout
  `Nothing`, donc la précision sur les classes rares y serait plus faible.
- Le pat n'est pas géré (aucune position de pat dans `assets/`).

## Pistes non réalisées

- Des caractéristiques qui donnent au réseau ce qui lui manque (simuler le blocage et la capture de la pièce qui donne l'échec), plutôt que plus d'entraînement, pour réduire la confusion échec / échec et mat ; plus de positions (on en a utilisé 200 000 sur environ 720 000).
- Un planning de learning rate.
- Normalisation par batch, poids de classes.
- Benchmarks avec plusieurs graines et recherche aléatoire plus large.
- Le pat (demande des positions absentes de `assets/`).
- Parallélisme (le multiprocessing est déjà utilisé pour les benchmarks ; l'entraînement lui-même tourne dans un seul processus).

## Si on recommençait

On générerait nos propres positions étiquetées avec un moteur d'échecs (pour
inclure le pat et des proportions de classes réalistes), et on garderait un jeu
de test issu d'une autre source. On comparerait aussi l'encodage brut aux
caractéristiques ajoutées, et on fixerait la taille de batch dès le départ avec
la recherche aléatoire. Une couche convolutive sur l'échiquier serait une
alternative naturelle, au prix d'une rétropropagation plus complexe.

## Données et vie privée (RGPD)

Les données sont des positions d'échecs (chaînes FEN) et leurs étiquettes. Elles
ne contiennent aucune donnée personnelle. Le projet ne collecte ni n'envoie
rien.

## Lancer les tests

Depuis la racine du dépôt (après `make`, car les tests de l'analyseur utilisent
l'exécutable `my_torch_analyzer`) :

```sh
python -m unittest discover -s tests -v
```

Voir [docs/test.md](docs/test.md) pour les commandes de test et la couverture,
[docs/analyser.md](docs/analyser.md) pour le fonctionnement de l'analyseur, et
[docs/nn.md](docs/nn.md) pour les composants du réseau de neurones.
