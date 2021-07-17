from typing import Type, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.core import Agent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from gym import Env

from games.connect4.env import Connect4Env, Connect4SecondPlayerEnv, winning_combos
from games.connect4.processor import Connect4Processor


def get_dqn_agent(env: Env) -> Agent:
    """
    >>> env = Connect4Env()
    >>> agent = get_dqn_agent(env)
    >>> agent.layers[1].weights[0].shape
    TensorShape([27, 27])
    >>> agent.layers[2].weights[0].shape
    TensorShape([27, 9])
    """
    nb_actions = env.action_space.n

    model = Sequential([
        Flatten(input_shape=(1,) + env.observation_space.shape),
        Dense(len(list(winning_combos())), activation='relu'),
        Dense(nb_actions, activation='linear'),
    ])

    memory = SequentialMemory(limit=50000, window_length=1)
    training_policy = EpsGreedyQPolicy(eps=0.2)
    test_policy = BoltzmannQPolicy()
    processor = Connect4Processor()
    dqn = DQNAgent(model=model,
                   processor=processor,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=100,
                   target_model_update=1e-2,
                   policy=training_policy,
                   test_policy=test_policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def train_agent(env: Env, agent: Agent, steps: int = 10000) -> Agent:
    agent.fit(env, nb_steps=steps, visualize=False, verbose=1)
    return agent


def train_against(trainee: Agent, trainee_env: Type[Env], opponent: Agent, steps: int = 10000) -> Env:
    trainee.training = True
    opponent.training = False
    # Comment the next line as we now use a test_policy (ie. when not in training) with some randomness.
    # opponent.training = True  # So that it still takes random choices occasionally when played against.
    env = trainee_env(get_opponent_action=lambda board: opponent.forward(opponent.processor.process_observation(board)))
    train_agent(env, trainee, steps)
    return env


def load_agents(path_base: str) -> Tuple[Agent, Env, Agent, Env]:
    path_ext = '.hdf5'
    env1 = Connect4Env()
    env2 = Connect4SecondPlayerEnv()
    agent1 = get_dqn_agent(env1)
    agent2 = get_dqn_agent(env2)
    agent1.load_weights(f'{path_base}-1{path_ext}')
    agent2.load_weights(f'{path_base}-2{path_ext}')
    return agent1, env1, agent2, env2


def save_agents(path_base: str, agent1: Agent, agent2: Agent) -> None:
    path_ext = '.hdf5'
    agent1.save_weights(f'{path_base}-1{path_ext}')
    agent2.save_weights(f'{path_base}-2{path_ext}')


def play(env: Env, agent: Agent) -> None:
    done = False
    observation = env.reset()
    agent.training = False
    step = 0
    while not done:
        print("Turn: {}".format(step + 1))
        processed_observation = agent.processor.process_observation(observation)
        action = agent.forward(processed_observation)
        observation, reward, done, info = env.step(action)
        env.render()
        print(f"{'Game over' if done else ''} Reward: {reward} {info}\n")
        step += 1
