

class GymEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.acton_dim = env.action_space.shape[0]

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    @property
    def count(self):
        count = self.env.spec.timestep_limit
        return count

    def render(self):
        self.env.render()


class EnvWrapper:
    def __init__(self, environment):
        self.env = environment

    def train(self, method):
        agent = method.train_agent()
        state = self.env.reset()
        for i in xrange(self.env.count):
            action = agent.action(state)
            next_state, reward, done = self.env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    def demo(self, method):
        agent = method.agent()
        state = self.env.reset()
        total_reward = 0
        for i in xrange(self.env.count):
            self.env.render()
            action = agent.action(state)  # direct action for test
            state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
