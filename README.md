# Tic-tac-toe (aka. noughts and crosses)
With Deep Q (DQN) reinforcement learning using gym and keras-rl

## Pre-requisites

Because I'm using an M1 Mac, I can't set this up using pipenv. I've had to install python 3.8 and tensorflow using miniforge.

So you're on your own re requirements – sorry!

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
python -m nac.play
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
