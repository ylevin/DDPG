import filter_env
import gym.wrappers as gw
from ddpg import *
import gc

gc.enable()

# ENV_NAME = 'MountainCarContinuous-v0'
ENV_NAME = 'Pendulum-v0'
EPISODES = 10000
TEST = 5
TEST_EVERY = 20
TEST_SINCE = 80


def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    mon = gw.Monitor(env, 'experiments/' + ENV_NAME, force=True)
    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        print "episode:", episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % TEST_EVERY == 0 and episode >= TEST_SINCE:
            total_reward = 0
            for i in xrange(TEST):
                print "Start test #{}".format(i)
                state = env.reset()
                for j in xrange(env.spec.timestep_limit):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
    # env.monitor.close()
    mon.close()


if __name__ == '__main__':
    main()
