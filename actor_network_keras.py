import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model

# Hyper Parameters
LEARNING_RATE = 1e-4
TAU = 0.001


class ActorNetwork:
    """docstring for ActorNetwork"""

    def __init__(self, sess, state_dim, action_dim, lr=LEARNING_RATE, tau=TAU):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        # create actor network
        self.model, self.state_input, self.action_output = \
            ActorNetwork.create_network(state_dim, action_dim)

        # create target actor network
        self.target_model, self.target_state_input, self.target_action_output = \
            ActorNetwork.create_network(state_dim, action_dim)

        # define training rules
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(
            self.action_output, self.model.trainable_weights, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(
            zip(self.parameters_gradients, self.model.trainable_weights))

        self.target_model.set_weights(self.model.get_weights())

        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def create_network(state_dim, action_dim):
        state_input = Input([state_dim])
        layer_1 = Dense(400, activation='relu')(state_input)
        layer_2 = Dense(300, activation='relu')(layer_1)
        action_output = Dense(action_dim, activation='tanh')(layer_2)
        model = Model([state_input], action_output)
        return model, state_input, action_output

    def update_target(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in xrange(len(target_model_weights)):
            target_model_weights[i] = model_weights[i] * self.tau + target_model_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_model_weights)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })

    def actions(self, state_batch):
        return self.model.predict_on_batch(state_batch)

    def action(self, state):
        # return self.model.predict([state])
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def target_actions(self, state_batch):
        return self.target_model.predict_on_batch(state_batch)


'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''
