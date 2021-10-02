import os
from games.connect4.env import Connect4Env, Connect4SecondPlayerEnv
from games.connect4.agent import train_against, train_agent, load_agents, get_dqn_agent, play, save_agents, test
from games.connect4.play_human import play_human
# from tensorflow.python.framework.ops import disable_eager_execution
# from tensorflow.python.compiler.mlcompute import set_mlc_device

if __name__ == '__main__':
    # As suggested by https://github.com/apple/tensorflow_macos/issues/268
    # Note this uses the GPU but is actually slightly slower than with CPU.
    # disable_eager_execution()
    # set_mlc_device(device_name='gpu')

    DEFAULT_WEIGHT_FILE_NAME = 'temp'
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    WORDS = ['']
    while WORDS[0] not in ('load', 'new', 'improve'):
        WORD = input("""
Type one of:
    - "load NAME" to load pre-trained agents from the named file (eg. "load weights")
    - "new X" to train new agents over X rounds (eg. "new 20")
    - "improve NAME X" to load pre-trained agents and continue training them (eg. "improve temp 5")
Your choice? """)
        WORDS = WORD.split(' ')

    if WORDS[0] == 'load':
        FILENAME = 'weights'
        try:
            FILENAME = WORDS[1]
        except IndexError:
            pass
        agent_1, env_1, agent_2, env_2 = load_agents(os.path.join(SCRIPT_PATH, 'weights', FILENAME))
    elif WORDS[0] == 'improve':
        INITIAL = 0
        ROUNDS = 10
        FILENAME = 'weights'
        try:
            FILENAME = WORDS[1]
            ROUNDS = int(WORDS[2])
        except (ValueError, IndexError):
            pass
        agent_1, env_1, agent_2, env_2 = load_agents(os.path.join(SCRIPT_PATH, 'weights', FILENAME))
    else:
        ROUNDS = 10
        try:
            ROUNDS = int(WORDS[1])
        except (ValueError, IndexError):
            pass

        env_1 = Connect4Env()
        agent_1 = get_dqn_agent(env_1)
        env_2 = Connect4SecondPlayerEnv()
        agent_2 = get_dqn_agent(env_2)

        print()
        print(f'Round 1 of {ROUNDS} (against random legal actions)')
        print('Training player 1')
        agent_1 = train_agent(env_1, agent_1)
        print('Training player 2')
        agent_2 = train_agent(env_2, agent_2)
        INITIAL = 1

    if WORDS[0] in ('improve', 'new'):
        for i in range(INITIAL, ROUNDS):
            old_env_1 = env_1
            old_env_2 = env_2
            print()
            print(f'Round {i + 1} of {ROUNDS}')
            print('Training player 1')
            env_1 = train_against(agent_1, Connect4Env, agent_2)
            print('Testing against previous player 2')
            test(old_env_1, agent_1)
            print('Testing against latest player 2')
            test(env_1, agent_1)
            print(f'Round {i + 1} of {ROUNDS}')
            print('Training player 2')
            env_2 = train_against(agent_2, Connect4SecondPlayerEnv, agent_1)
            print('Testing against previous player 1')
            test(old_env_2, agent_2)
            print('Testing against latest player 1')
            test(env_2, agent_2)

            print(f'Saving weights for trained agents (as {DEFAULT_WEIGHT_FILE_NAME})')
            save_agents(os.path.join(SCRIPT_PATH, 'weights', DEFAULT_WEIGHT_FILE_NAME), agent_1, agent_2)

    print("Play against themselves:")
    play(env_1, agent_1)
    play(env_2, agent_2)

    AGAIN = 'yes'
    while AGAIN.lower() != 'no':
        print("Play against you - you play first:")
        play_human(Connect4SecondPlayerEnv, agent_2)
        print("Play against you - you play second:")
        play_human(Connect4Env, agent_1)
        AGAIN = input('Play again ("no" to end)? ')
