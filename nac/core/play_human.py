import sys
from typing import Optional, Type
from gym.core import Env
from rl.core import Agent

from nac.core.types import Board, Action

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
    # Annoying that render is part of env, and not a property of Board.
    print("{}{}{}  012\n{}{}{}  345\n{}{}{}  678".format(*board.tolist()))
    user_input = ''
    while to_int(user_input) is None:
        user_input = input('Which square number (0-8)? ')
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
