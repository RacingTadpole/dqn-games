# The encoding of the board is based on
# https://github.com/mahowald/tictactoe/blob/master/tictactoe/env.py
from typing import Callable, Literal, Tuple, Generator, List
import numpy as np
from gym import spaces, Env
from gym.utils import seeding

from .types import Action, Board

WIDTH = 7
HEIGHT = 6
NUM_POSITIONS = WIDTH * HEIGHT
MARKS = ['•', 'X', 'O']

def winning_combos() -> Generator[List[int], None, None]:
    """
    >>> len(list(winning_combos()))
    69
    """
    # Horizontal runs
    for row in range(HEIGHT):
        for column in range(WIDTH - 3):
            base = row * WIDTH + column
            yield [base, base + 1, base + 2, base + 3]
    # Vertical runs
    for row in range(HEIGHT - 3):
        for column in range(WIDTH):
            base = row * WIDTH + column
            yield [base, base + 1 * WIDTH, base + 2 * WIDTH, base + 3 * WIDTH]
    # Diagonal runs sloping down
    for row in range(HEIGHT - 3):
        for column in range(WIDTH - 3):
            base = row * WIDTH + column
            yield [base, base + 1 * WIDTH + 1, base + 2 * WIDTH + 2, base + 3 * WIDTH + 3]
    # Diagonal runs sloping up
    for row in range(3, HEIGHT):
        for column in range(WIDTH - 3):
            base = row * WIDTH + column
            yield [base, base - 1 * WIDTH + 1, base - 2 * WIDTH + 2, base - 3 * WIDTH + 3]

class Connect4Env(Env):
    """
    Connect 4.
    Width 7, height 6 board looks like:
    [0, 1, 2, 3, 4, 5, 6,
     7, 8, 9,10,11,12,13,
     ...
    35,36,37,38,39,40,41]
    """
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiBinary(NUM_POSITIONS * 3)
    action_space = spaces.Discrete(WIDTH)

    def __init__(self, get_opponent_action: Callable[[Board], Action]=None) -> None:
        super().__init__()
        self.np_random = seeding.np_random(1)
        self.action_space.seed(1)
        default_get_action = lambda _: self.action_space.sample()
        self.get_opponent_action = get_opponent_action or default_get_action
        # Both of these encode the state, and are mutable.
        self.current_player: Literal[0, 1] = 0
        self.board = np.zeros(NUM_POSITIONS, dtype="int")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self) -> Board:
        """
        >>> env = Connect4Env()
        >>> env.reset().shape
        (42,)
        """
        self.current_player = 0
        self.board = np.zeros(NUM_POSITIONS, dtype=np.int8)
        return self.board

    def _is_legal(self, action: Action) -> bool:
        """
        >>> env = Connect4Env()
        >>> obs = env.reset()
        >>> for _ in range(4):
        ...     _ = env.step(0)
        ...     _ = env.step(4)
        >>> env.render()
        • • • • X • •
        X • • • O • •
        X • • • X • •
        X • O • X • •
        O • O • O • •
        X • O O X O •
        >>> env._is_legal(0), env._is_legal(2), env._is_legal(4)
        (True, True, False)
        """
        # Cannot place chip into a column that is full.
        return self.board[action] == 0

    def _can_other_player_win_next(self) -> bool:
        """
        >>> env = Connect4Env()
        >>> obs = env.reset()
        >>> for a in [v for v in range(0, WIDTH, 2) for _ in range(2)]:
        ...     __ = env.drop_chip(a)
        >>> env.current_player = 1
        >>> env._can_other_player_win_next()
        False
        >>> env.current_player = 0
        >>> env.drop_chip(0)
        >>> env.current_player = 1
        >>> env._can_other_player_win_next()
        True
        """
        for combo in winning_combos():
            # Looking for three spaces in the combo claimed by the other player, with one empty,
            # and the spaces below them are all occupied (or it's the bottom row).
            if (self.board[list(combo)] == 2 - self.current_player).sum() == 3 \
                    and (self.board[list(combo)] == 0).sum() == 1 \
                    and (self.board[list(x + WIDTH for x in combo if x < NUM_POSITIONS - WIDTH)] > 0).all():
                return True
        return False

    def _has_current_player_won(self) -> bool:
        """
        >>> env = Connect4Env()
        >>> obs = env.reset()
        >>> for a in (1, 3, 4):
        ...     _ = env.step(a)
        >>> env._has_current_player_won()
        False
        >>> _ = env.step(2)
        >>> env._has_current_player_won()
        True
        """
        for combo in winning_combos():
            if (self.board[list(combo)] == self.current_player + 1).all():
                return True
        return False

    def _is_board_full_next(self) -> bool:
        return (self.board != 0).sum() >= NUM_POSITIONS - 1

    def drop_chip(self, action: Action) -> None:
        """
        >>> env = Connect4Env()
        >>> obs = env.reset()
        >>> env.drop_chip(3)
        >>> env.render()
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • X • • •
        >>> for a in (0, 4, 1, 1, 5, 0, 2, 1, 1, 4):
        ...     _ = env.drop_chip(a)
        >>> env.render()
        • • • • • • •
        • • • • • • •
        • X • • • • •
        • X • • • • •
        X X • • X • •
        X X X X X X •
        """
        if action > WIDTH:
            raise Exception('Illegal action')
        for row in range(HEIGHT - 1, -1, -1):
            if self.board[row * WIDTH + action] == 0:
                self.board[row * WIDTH + action] = self.current_player + 1
                break


    def step(self, action: Action) -> Tuple[Board, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Mutates self.board and self.current_player.
        Returns (new observation, reward, done, info).

        >>> env = Connect4Env()
        >>> obs = env.reset()
        >>> _ = env.step(2)
        >>> env.render()
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • X • • O •
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

        # drop the chip to the bottom of this column
        self.drop_chip(action)

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
            reward = 0.5
            info = {
                "state": "done",
                "reason": "Players have tied (or are about to)",
            }
            done = True

        # move to the next player
        if not done:
            self.current_player = 1 - self.current_player
            opponent_action = self.get_opponent_action(self.board)
            counter = 0
            while not self._is_legal(opponent_action) and counter < 100:
                opponent_action = self.get_opponent_action(self.board) if counter < 25 else self.action_space.sample()
                counter += 1
            if counter == 100:
                done = True
            else:
                self.drop_chip(opponent_action)
                # And return to the original player's turn.
                self.current_player = 1 - self.current_player

        return self.board, reward, done, info

    def render(self, mode: str = "human"):
        for row in range(HEIGHT):
            print(*[MARKS[x] for x in self.board[row * WIDTH: (row + 1) * WIDTH].tolist()])


class Connect4SecondPlayerEnv(Connect4Env):
    """
    Noughts and crosses where you play second.
    The only difference is the reset function, which starts you after the opponent has
    already taken a turn.
    """
    def reset(self) -> Board:
        """
        >>> env = Connect4SecondPlayerEnv()
        >>> _ = env.reset()
        >>> env.render()
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • • •
        • • • • • X •
        """
        super().reset()
        opponent_action = self.get_opponent_action(self.board)
        self.drop_chip(opponent_action)
        self.current_player = 1 - self.current_player
        return self.board
