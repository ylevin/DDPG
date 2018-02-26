import filter_env
import gym.wrappers as gw
import gym
from evolution import *
from env_wrapper import *
import gc

gc.enable()

# ENV_NAME = 'MountainCarContinuous-v0'
ENV_NAME = 'Pendulum-v0'
EPISODES = 1000000
TEST = 5
TEST_EVERY = 1000
TEST_SINCE = 5000


def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env = GymEnvAdapter(env)
    wrapper = EnvWrapper(env)
    method = DifferentialEvolutionMethod(env.state_dim, env.acton_dim)

    for episode in xrange(EPISODES):
        print "episode:", episode
        # Train
        wrapper.train(method)

        # Testing:
        if episode % TEST_EVERY == 0 and episode >= TEST_SINCE:
            total_reward = 0
            for i in xrange(TEST):
                print "Start test #{}".format(i)
                total_reward += wrapper.demo(method)
            ave_reward = total_reward / TEST
            print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward


if __name__ == '__main__':
    main()
