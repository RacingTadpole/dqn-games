# Reinforcement learning games

Some games with Deep Q (DQN) reinforcement learning, using gym and keras-rl.

## Pre-requisites

Because I'm using an M1 Mac, I can't set this up using pipenv. I've had to install python 3.8 and tensorflow using miniforge, with (not reconfirmed):

```
brew install miniforge
conda env create --file=environment.yml --name tf_m1
conda init tf_m1
conda activate tf_m1
pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
conda install -c conda-forge tqdm
conda install -c conda-forge mypy
conda install -c conda-forge pytest
conda install -c conda-forge pylint
pip install keras-rl "gym[atari]"
```

Then to enter the correct environment, use 

```
conda activate tf_m1
```

## Tic-tac-toe (aka. noughts and crosses)

## Comments

This is a two player game, so I have modelled the opponent as part of the environment, and defined two environments:
one where the AI agent goes first, and one where it goes second.

The training starts by playing each agent against randomly chosen moves, and then bootstraps by playing each agent
against the a frozen version of the other agent.

I have arbitrarily chosen a single 27-node hidden layer for the neural network.
It would be interesting to try other configurations.

The saved players are pretty good, but definitely make mistakes.

## Play against the AI

```
python -m games.nac.play
```

Eg.
```
Type "load" to load pre-trained agents, or "new" to train new ones: load
Playing against themselves:
...
Playing against you - you play second:
Turn: 1
•••  012
•••  345
••X  678
Which square number (0-8)? 7
Turn: 2
•••  012
•X•  345
•OX  678
Which square number (0-8)? 0
Turn: 3
O•X  012
•X•  345
•OX  678
Which square number (0-8)? 6
Turn: 4
O•X  012
XX•  345
OOX  678
Which square number (0-8)? 5
Turn: 5
OXX
XXO
OOX
Game over: Players have tied (or are about to)...
```
