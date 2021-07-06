import numpy as np
from rl.processors import Processor

from nac.core.types import BoardForDqn, Board


class NacProcessor(Processor):
    def process_observation(self, observation: Board) -> BoardForDqn:
        """
        Processes the observation as obtained from the environment for use in an agent and
        returns it.
        27 input neurons.
        The first 3 are square 1, the next 3 are square 2 etc.
        Every neuron is 1 or 0.
        The first of the set of three indicates whether this square is free or not;
        the second indicates whether the square is occupied by your opponent or not.
        Only one in every 3 neurons will have a 1, the other two will have a 0.

        >>> board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        >>> p = NacProcessor()
        >>> p.process_observation(board)
        array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
               0, 0, 1, 0, 0], dtype=int8)
        """
        return np.eye(3, dtype=np.int8)[observation].reshape(-1)
