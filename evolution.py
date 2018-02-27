from keras.layers import Input, Dense
from keras.models import Model

import tensorflow as tf

import keras.backend as K

from random import sample

import numpy as np

F = 1.0
CROSSOVER_PROBABILITY = 0.3
MUTATION_PROBABILITY = 0.1
POPULATION_SIZE = 50
ATTEMPTS_SIZE = 10
EPSILON = 0.05


class DifferentialEvolutionMethod:
    def __init__(self, env, population_size=POPULATION_SIZE, attempts_size=ATTEMPTS_SIZE,
                 actor_network_factory=None):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.actor_network_factory = \
            actor_network_factory if actor_network_factory is not None \
            else DifferentialEvolutionMethod.create_network
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.population_size = population_size
        self.attempts_size = attempts_size
        self.population = []
        for i in xrange(population_size):
            self.population += [self.actor_network_factory(self.state_dim, self.action_dim)]

        self.sess.run(tf.global_variables_initializer())

        self.best_score = None
        self.best_n = 0

    def test_model(self, model):
        agent = DifferentialEvolutionTrainAgent(model)
        state = self.env.reset()

        for i in xrange(self.env.steps_count):
            action = agent.action(state)
            next_state, reward, done, _ = self.env.step(action)
            agent.add_reward(reward)
            state = next_state
            if done:
                break

        return agent.reward

    def train(self):
        for i in xrange(self.population_size):
            print "select {}".format(i)
            original_model = self.population[i]
            new_model = self.crossover(self.population[i])
            sum_reward_original = 0
            sum_reward_new = 0

            for _ in xrange(self.attempts_size):
                sum_reward_original += self.test_model(original_model)
                sum_reward_new += self.test_model(new_model)

            if sum_reward_new > sum_reward_original:
                print "new net is better: {} vs {}, let's replace!".format(sum_reward_new, sum_reward_original)
                self.population[i] = new_model
            else:
                print "new net lost: {} vs {}".format(sum_reward_new, sum_reward_original)

            best_reward = max(sum_reward_new, sum_reward_original)

            if self.best_score is None or best_reward > self.best_score:
                self.best_score = sum_reward_new
                self.best_n = i

    @staticmethod
    def create_network(state_dim, action_dim):
        state_input = Input([state_dim])
        layer_1 = Dense(100, activation='relu', bias_initializer='glorot_uniform')(state_input)
        layer_2 = Dense(50, activation='relu', bias_initializer='glorot_uniform')(layer_1)
        action_output = Dense(action_dim, activation='tanh')(layer_2)
        model = Model([state_input], action_output)
        return model

    def make_new_population(self, selected):
        new_population = []
        for i in xrange(self.population_size):
            print "crossover {}".format(i)
            new_population += [self.crossover(selected)]
        return new_population

    def crossover(self, original):
        sample_pop = sample(self.population, 4)

        if original in sample_pop:
            sample_pop.remove(original)

        original_net = original.get_weights()
        net1 = sample_pop[0].get_weights()
        net2 = sample_pop[1].get_weights()
        net3 = sample_pop[2].get_weights()

        for i in xrange(len(original_net)):
            need_cross = np.random.uniform(0, 1, size=original_net[i].shape) < CROSSOVER_PROBABILITY
            original_net[i] = np.where(need_cross, net1[i] + F * (net2[i] - net3[i]), original_net[i])

        new_individual = self.actor_network_factory(self.state_dim, self.action_dim)
        new_individual.set_weights(original_net)

        return new_individual

    def demo(self):
        agent = DifferentialEvolutionAgent(self.population[self.best_n])
        state = self.env.reset()
        total_reward = 0

        for i in xrange(self.env.steps_count):
            self.env.render()
            action = agent.action(state)  # direct action for test
            new_state, reward, done, _ = self.env.step(action)
            state = new_state
            total_reward += reward
            if done:
                break

        return total_reward


class DifferentialEvolutionAgent:
    def __init__(self, model):
        self.model = model

    def action(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))[0]


class DifferentialEvolutionTrainAgent(DifferentialEvolutionAgent):
    def __init__(self, model):
        DifferentialEvolutionAgent.__init__(self, model)
        self.reward = 0

    def add_reward(self, reward):
        self.reward = EPSILON * reward + self.reward * (1 - EPSILON)
