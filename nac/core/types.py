from typing import NewType
import numpy as np

Action = NewType('Action', int)
Board = NewType('Board', np.ndarray)  # 1D array of shape (9,)
BoardForDqn = NewType('BoardForDqn', np.ndarray)  # 1D array of shape (27,)
