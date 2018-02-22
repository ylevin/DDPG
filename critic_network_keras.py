import tensorflow as tf
from keras.layers import Input, Dense, concatenate
from keras.models import Model

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01


class CriticNetwork:
    """docstring for CriticNetwork"""

    def __init__(self, sess, state_dim, action_dim):
        self.time_step = 0
        self.sess = sess
        # create q network
        self.model, self.state_input, self.action_input, self.q_value_output = \
            self.create_q_network(state_dim, action_dim)
        self.net = self.model.trainable_weights

        # create target q network (the same structure with q network)
        self.target_model, _, _, _ = self.create_q_network(state_dim, action_dim)

        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)
        self.target_model.set_weights(self.model.get_weights())

        # initialization
        self.sess.run(tf.initialize_all_variables())

    @staticmethod
    def create_q_network(state_dim, action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = Input([state_dim])
        action_input = Input([action_dim])

        layer_1 = Dense(layer1_size, activation='relu')(state_input)
        merge_layer = concatenate([layer_1, action_input])
        layer_2 = Dense(layer2_size, activation='relu')(merge_layer)
        q_value_output = Dense(1, activation='linear')(layer_2)

        model = Model([state_input, action_input], q_value_output)

        return model, state_input, action_input, q_value_output

    def update_target(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in xrange(len(target_model_weights)):
            target_model_weights[i] = model_weights[i] * TAU + target_model_weights[i] * (1 - TAU)
        self.target_model.set_weights(target_model_weights)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.target_model.predict_on_batch([state_batch, action_batch])

    def q_value(self, state_batch, action_batch):
        return self.model.predict_on_batch([state_batch, action_batch])


'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self, time_step):
        print 'save critic-network...', time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step=time_step)
'''
