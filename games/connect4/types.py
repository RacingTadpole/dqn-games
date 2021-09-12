from typing import NewType
import numpy as np

Action = NewType('Action', int)  # number from 0 to width - 1 (6).
Board = NewType('Board', np.ndarray)  # 1D array of shape (width (7) * height (6) = 42,)
BoardForDqn = NewType('BoardForDqn', np.ndarray)  # 1D array of shape (width * height * 3 = 126,)
