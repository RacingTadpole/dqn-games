import os
from games.connect4.env import Connect4Env, Connect4SecondPlayerEnv
from games.connect4.agent import train_against, train_agent, load_agents, get_dqn_agent, play, save_agents
from games.connect4.play_human import play_human

if __name__ == '__main__':
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
        play(env_1, agent_1)
        print('Training player 2')
        agent_2 = train_agent(env_2, agent_2)
        play(env_2, agent_2)
        INITIAL = 1

    if WORDS[0] in ('improve', 'new'):
        for i in range(INITIAL, ROUNDS):
            print()
            print(f'Round {i + 1} of {ROUNDS}')
            print('Training player 1')
            env_1 = train_against(agent_1, Connect4Env, agent_2)
            play(env_1, agent_1)
            print('Training player 2')
            env_2 = train_against(agent_2, Connect4SecondPlayerEnv, agent_1)
            play(env_2, agent_2)

            print('Saving weights for trained agents (as temp)')
            save_agents(os.path.join(SCRIPT_PATH, 'weights', 'temp'), agent_1, agent_2)

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
