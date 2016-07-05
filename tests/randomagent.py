# The world's simplest agent for testing environment
class RandomAgent(object):


    def __init__(self, action_space):
        self.action_space = action_space


    def act(self, observation, reward, done):
        return self.action_space.sample()
