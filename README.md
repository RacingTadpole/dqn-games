# Noughts and Crosses with reinforcement learning using gym and keras-rl

## Pre-requisites

Because I'm using an M1 Mac, I can't set this up using pipenv. So you're on your own re requirements.

## Play against the AI

```
python -m nac.play
```

Eg.
```
Type "load" to load pre-trained agents, or "new" to train new ones: load
Playing against themselves:
...
Playing against you - you play first:
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
