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
    def __init__(self, state_dim, action_dim, population_size=POPULATION_SIZE, attempts_size=ATTEMPTS_SIZE,
                 actor_network_factory=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_network_factory = \
            actor_network_factory if actor_network_factory is not None \
            else DifferentialEvolutionMethod.create_network
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.population_size = population_size
        self.attempts_size = attempts_size
        self.population = []
        for i in xrange(population_size):
            self.population += [self.actor_network_factory(state_dim, action_dim)]

        self.sess.run(tf.global_variables_initializer())
        self.train_agents = self.train_agent_generator()

        self.best_score = None
        self.best_individual = None

    def train_agent(self):
        return self.train_agents.next()

    def agent(self):
        return DifferentialEvolutionAgent(self.best_individual, self.sess)

    def train_agent_generator(self):
        while True:
            for i in xrange(self.population_size):
                print "select {}".format(i)
                new_individual = self.crossover(self.population[i])
                original_agent = DifferentialEvolutionTrainAgent(self.population[i], self.sess)
                new_agent = DifferentialEvolutionTrainAgent(new_individual, self.sess)
                sum_reward_original = 0
                sum_reward_new = 0

                for _ in xrange(self.attempts_size):
                    yield original_agent
                    sum_reward_original += original_agent.reward

                for _ in xrange(self.attempts_size):
                    yield new_agent
                    sum_reward_new += new_agent.reward

                if sum_reward_new > sum_reward_original:
                    print "new net is better: {} vs {}, let's replace!".format(sum_reward_new, sum_reward_original)
                    self.population[i] = new_individual
                else:
                    print "new net lost: {} vs {}".format(sum_reward_new, sum_reward_original)

                if self.best_score is None or sum_reward_new > self.best_score:
                    self.best_score = sum_reward_new
                    self.best_individual = new_individual

                if self.best_score is None or sum_reward_original > self.best_score:
                    self.best_score = sum_reward_original
                    self.best_individual = self.population[i]

    @staticmethod
    def create_network(state_dim, action_dim):
        state_input = Input([state_dim])
        layer_1 = Dense(10, activation='relu', bias_initializer='glorot_uniform')(state_input)
        layer_2 = Dense(10, activation='relu', bias_initializer='glorot_uniform')(layer_1)
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


class DifferentialEvolutionAgent:
    def __init__(self, model, sess):
        self.model = model
        self.sess = sess
        self.state_input = model.input
        self.action_output = model.output

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]


class DifferentialEvolutionTrainAgent(DifferentialEvolutionAgent):
    def __init__(self, model, sess):
        DifferentialEvolutionAgent.__init__(self, model, sess)
        self.reward = 0

    def perceive(self, state, action, reward, next_state, done):
        self.reward = EPSILON * reward + self.reward * (1 - EPSILON)
