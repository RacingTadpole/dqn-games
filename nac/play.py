from nac.core.env import NacEnv, NacSecondPlayerEnv
from nac.core.agent import train_against, train_agent, load_agents, get_dqn_agent, play
from nac.core.play_human import play_human

if __name__ == '__main__':
    WORD = ''
    while WORD not in ('load', 'new'):
        WORD = input('Type "load" to load pre-trained agents, or "new" to train new ones: ')
    if WORD == 'load':
        agent_1, env_1, agent_2, env_2 = load_agents('weights')
    else:
        env_1 = NacEnv()
        agent_1 = get_dqn_agent(env_1)
        agent_1 = train_agent(env_1, agent_1)
        play(env_1, agent_1)

        env_2 = NacSecondPlayerEnv()
        agent_2 = get_dqn_agent(env_2)
        agent_2 = train_agent(env_2, agent_2)
        play(env_2, agent_2)

        for _ in range(10):
            env_1 = train_against(agent_1, NacEnv, agent_2)
            env_2 = train_against(agent_2, NacSecondPlayerEnv, agent_1)

    print("Playing against themselves:")
    play(env_1, agent_1)
    play(env_2, agent_2)

    print("Playing against you - you play second:")
    play_human(NacEnv, agent_1)
    print("Playing against you - you play first:")
    play_human(NacSecondPlayerEnv, agent_2)
