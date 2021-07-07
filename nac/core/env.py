# The encoding of the board is based on
# https://github.com/mahowald/tictactoe/blob/master/tictactoe/env.py
from typing import Callable, Literal, Tuple
import numpy as np
from gym import spaces, Env
from gym.utils import seeding

from .types import Action, Board


MARKS = ['•', 'X', 'O']
class NacEnv(Env):
    """
    Noughts and crosses.
    Board looks like:
    [0, 1, 2,
     3, 4, 5,
     6, 7, 8]
    """
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiBinary(9 * 3)
    action_space = spaces.Discrete(9)

    winning_combos = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def __init__(self, get_opponent_action: Callable[[Board], Action]=None) -> None:
        super().__init__()
        self.np_random = seeding.np_random(1)
        self.action_space.seed(1)
        default_get_action = lambda _: self.action_space.sample()
        self.get_opponent_action = get_opponent_action or default_get_action
        # Both of these encode the state, and are mutable.
        self.current_player: Literal[0, 1] = 0
        self.board = np.zeros(9, dtype="int")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self) -> Board:
        """
        >>> env = NacEnv()
        >>> env.reset()
        array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
        """
        self.current_player = 0
        self.board = np.zeros(9, dtype=np.int8)
        return self.board

    def _is_legal(self, action: Action) -> bool:
        """
        >>> env = NacEnv()
        >>> obs = env.reset()
        >>> for a in (0, 4, 1):
        ...     _ = env.step(a)
        >>> env._is_legal(3), env._is_legal(4)
        (True, False)
        """
        return self.board[action] == 0

    def _can_other_player_win_next(self) -> bool:
        """
        >>> env = NacEnv()
        >>> obs = env.reset()
        >>> for a in (1, 4):
        ...     _ = env.step(a)
        >>> env.current_player
        0
        >>> env.render()
        OXO
        •X•
        •••
        >>> env._can_other_player_win_next()
        False
        >>> for a in (8, 5):
        ...     _ = env.step(a)
        >>> env.render()
        OXO
        •XX
        O•X
        >>> env._can_other_player_win_next()
        True
        >>> env.board = np.array([2, 1, 0, 2, 1, 1, 1, 2, 2])
        >>> env.current_player = 1
        >>> env._can_other_player_win_next()
        True
        """
        for combo in self.winning_combos:
            # Looking for two spaces in the combo claimed by the other player, and one empty space.
            if (self.board[list(combo)] == 2 - self.current_player).sum() == 2 \
                    and (self.board[list(combo)] == 0).sum() == 1:
                return True
        return False

    def _has_current_player_won(self) -> bool:
        """
        >>> env = NacEnv()
        >>> obs = env.reset()
        >>> for a in (3, 4):
        ...     _ = env.step(a)
        >>> env._has_current_player_won()
        False
        >>> _ = env.step(5)
        >>> env.render()
        O•O
        XXX
        •••
        >>> env._has_current_player_won()
        True
        """
        for combo in self.winning_combos:
            if (self.board[list(combo)] == self.current_player + 1).all():
                return True
        return False

    def _is_board_full_next(self) -> bool:
        """
        >>> env = NacEnv()
        >>> obs = env.reset()
        >>> for a in (0, 4, 1, 3, 5, 8, 2):
        ...     _ = env.step(a)
        >>> env._is_board_full_next()
        False
        >>> _ = env.step(6)
        >>> env._is_board_full_next()
        True
        """
        return (self.board != 0).sum() >= 8

    def step(self, action: Action) -> Tuple[Board, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Mutates self.board and self.current_player.
        Returns (new observation, reward, done, info).

        >>> env = NacEnv()
        >>> obs = env.reset()
        >>> env.step(2)
        (array([2, 0, 1, 0, 0, 0, 0, 0, 0], dtype=int8), 0, False, {'state': 'in progress'})
        """
        info = {"state": "in progress"}
        reward = 0
        done = False

        # check if it's an illegal move
        if not self._is_legal(action):
            reward = -10  # illegal moves are really bad
            info = {"state": "done", "reason": "Illegal move"}
            done = True
            return self.board, reward, done, info

        self.board[action] = self.current_player + 1

        if self._has_current_player_won():
            reward = 1
            info = {
                "state": "done",
                "reason": "Player {} has won".format(self.current_player + 1),
            }
            done = True

        elif self._can_other_player_win_next():
            reward = -2
            info = {
                "state": "done",
                "reason": "Player {} will win".format(1 - self.current_player + 1),
            }
            done = True

        # check if the board is full now or on the next turn, which is a tie.
        elif self._is_board_full_next():
            reward = 0
            info = {
                "state": "done",
                "reason": "Players have tied (or are about to)",
            }
            done = True

        # move to the next player
        if not done:
            self.current_player = 1 - self.current_player
            opponent_action = self.get_opponent_action(self.board)
            while not self._is_legal(opponent_action):
                opponent_action = self.get_opponent_action(self.board)
            self.board[opponent_action] = self.current_player + 1
            # And return to the original player's turn.
            self.current_player = 1 - self.current_player

        return self.board, reward, done, info

    def render(self, mode: str = "human"):
        print("{}{}{}\n{}{}{}\n{}{}{}".format(*[MARKS[x] for x in self.board.tolist()]))


class NacSecondPlayerEnv(NacEnv):
    """
    Noughts and crosses where you play second.
    The only difference is the reset function, which starts you after the opponent has
    already taken a turn.
    """
    def reset(self) -> Board:
        """
        >>> env = NacSecondPlayerEnv()
        >>> env.reset()
        array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
        """
        super().reset()
        opponent_action = self.get_opponent_action(self.board)
        self.board[opponent_action] = self.current_player + 1
        self.current_player = 1 - self.current_player
        return self.board
