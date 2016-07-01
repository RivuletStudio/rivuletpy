
from rivuletpy.rivuletenv import RivuletEnv, RandomAgent 

if __name__ == '__main__':
    env = RivuletEnv(imgpath='tests/data/test-small.tif', swcpath='tests/data/test-small.swc', cached=False)
    env.reset()
    agent = RandomAgent(env.action_space)
    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            print('epoch %d step %d' % (i, j), end='\r')
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
