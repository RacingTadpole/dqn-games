import os
from games.nac.env import NacEnv, NacSecondPlayerEnv
from games.nac.agent import train_against, train_agent, load_agents, get_dqn_agent, play, save_agents
from games.nac.play_human import play_human

if __name__ == '__main__':
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    WORDS = ['']
    while WORDS[0] not in ('load', 'new'):
        WORD = input("""
Type:
    - "load NAME" to load pre-trained agents from the named file (eg. "load weights"), or
    - "new X" to train new agents over X rounds (eg. "new 20")
Your choice? """)
        WORDS = WORD.split(' ')

    if WORDS[0] == 'load':
        FILENAME = 'weights'
        try:
            FILENAME = WORDS[1]
        except IndexError:
            pass
        agent_1, env_1, agent_2, env_2 = load_agents(os.path.join(SCRIPT_PATH, 'weights', FILENAME))
    else:
        ROUNDS = 10
        try:
            ROUNDS = int(WORDS[1])
        except (ValueError, IndexError):
            pass

        env_1 = NacEnv()
        agent_1 = get_dqn_agent(env_1)
        env_2 = NacSecondPlayerEnv()
        agent_2 = get_dqn_agent(env_2)

        print()
        print(f'Round 1 of {ROUNDS}')
        print('Training player 1')
        agent_1 = train_agent(env_1, agent_1)
        play(env_1, agent_1)
        print('Training player 2')
        agent_2 = train_agent(env_2, agent_2)
        play(env_2, agent_2)

        for i in range(ROUNDS - 1):
            print()
            print(f'Round {i + 2} of {ROUNDS}')
            print('Training player 1')
            env_1 = train_against(agent_1, NacEnv, agent_2)
            play(env_1, agent_1)
            print('Training player 2')
            env_2 = train_against(agent_2, NacSecondPlayerEnv, agent_1)
            play(env_2, agent_2)

        print('Saving weights for trained agents')
        save_agents(os.path.join(SCRIPT_PATH, 'weights', 'temp'), agent_1, agent_2)

    print("Play against themselves:")
    play(env_1, agent_1)
    play(env_2, agent_2)

    print("Play against you - you play first:")
    play_human(NacSecondPlayerEnv, agent_2)
    print("Play against you - you play second:")
    play_human(NacEnv, agent_1)
