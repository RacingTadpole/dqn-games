import sys
from typing import Optional, Type
from gym.core import Env
from rl.core import Agent

from games.connect4.types import Board, Action
from games.connect4.env import MARKS, WIDTH, HEIGHT

def to_int(s: str) -> Optional[int]:
    """
    >>> to_int('foo')
    >>> to_int('4')
    4
    >>> to_int('4.6')
    """
    try:
        return int(s)
    except ValueError:
        return None


def get_human_action(board: Board) -> Action:
    print(' '.join([f'{i}' for i in range(WIDTH)]))
    # Would be nice if this wasn't repeated from env.
    for row in range(HEIGHT):
        print(*[MARKS[x] for x in board[row * WIDTH: (row + 1) * WIDTH].tolist()])

    user_input = ''
    while to_int(user_input) is None or to_int(user_input) > WIDTH - 1:
        user_input = input(f'Which column (0-{WIDTH - 1})? ')
        if user_input == '':
            sys.exit()
    return to_int(user_input)


def play_human(env_class: Type[Env], agent: Agent) -> None:
    env = env_class(get_opponent_action=get_human_action)
    done = False
    observation = env.reset()
    agent.training = False
    step = 0
    while not done:
        print("Turn: {}".format(step + 1))
        processed_observation = agent.processor.process_observation(observation)
        action = agent.forward(processed_observation)
        observation, _, done, info = env.step(action)
        step += 1
    env.render()
    print(f"Game over: {info['reason']}\n")
