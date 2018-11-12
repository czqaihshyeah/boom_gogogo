'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from MyAgent.NetAgent import *
import tensorflow as tf
import joblib

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    ##############
    sess = tf.InteractiveSession()
    params = joblib.load('model_3_policy')


    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.PlayerAgent(),
        # NetAgent(sess, params),
        #TODO 建立一个镜像
        agents.DockerAgent("pommerman/netagent1", port=12305),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            # print(state[1])
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        # print(state)
        print(done)
        print('Episode {} finished'.format(i_episode))
        print(reward)
    env.close()


if __name__ == '__main__':
    main()
