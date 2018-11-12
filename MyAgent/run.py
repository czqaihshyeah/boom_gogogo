"""Implementation of a simple deterministic agent using Docker."""
from NetAgent import *
from pommerman.runner import DockerAgentRunner
import tensorflow as tf
import joblib


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self,sess, params):
        self._agent = NetAgent(sess, params)

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    sess = tf.InteractiveSession()
    path = 'model_3_policy'
    params = joblib.load(path)
    agent = MyAgent(sess, params)
    agent.run()


if __name__ == "__main__":
    main()