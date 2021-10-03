import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ScaleLayer(keras.layers.Layer):
    """
    Kares layer subclass to scale raw sigmoid output to appropriate value
    """
    def __init__(self, action_range, min_action):
        super(ScaleLayer, self).__init__()
        self.action_range = tf.Variable(action_range, trainable = False, dtype = tf.float32)
        self.min_action = tf.Variable(min_action, trainable = False, dtype = tf.float32)

    def __call__(self, input):
        return tf.math.multiply(input, self.action_range) + self.min_action

class NoiseLayer(keras.layers.Layer):
    """
    Keras layer subclass to apply exploration noise to raw actions
    """
    def __init__(self):
        super(NoiseLayer, self).__init__()

    def __call__(self, input, noise):
        return tf.math.add(input, noise)

class CriticNetwork(keras.Model):
    """
    Define 2 layer dense critic network
    """
    def __init__(
        self, 
        layer_1_dims: int = 512, 
        layer_2_dims: int = 512,
        name: str = 'critic', 
        checkpoint_dir: str = 'tmp/checkpoints/ddpg'):

        super(CriticNetwork, self).__init__()
        self.layer_1_dims = layer_1_dims
        self.layer_2_dims = layer_2_dims

        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            self.model_name + '.h5'
        )

        initializer = keras.initializers.VarianceScaling(
            scale = 0.3333,
            mode = 'fan_in',
            distribution= 'uniform'
        )

        final_initializer = keras.initializers.RandomUniform(minval = -0.003, maxval=0.003)

        self.layer_1_dense = Dense(
            self.layer_1_dims, 
            activation = 'relu',
            kernel_initializer= initializer)
        self.layer_2_dense = Dense(
            self.layer_2_dims, 
            activation = 'relu',
            kernel_initializer= initializer)
        self.q = Dense(
            1, 
            activation = None,
            kernel_initializer = final_initializer,
            bias_initializer = final_initializer)

    def call(self, state: tf.Tensor, action: tf.Tensor):
        action_value = self.layer_1_dense(tf.concat([state, action], axis = 1))
        action_value = self.layer_2_dense(action_value)

        q = self.q(action_value)

        return q
        
class ActorNetwork(keras.Model):
    """
    Define 2 dense layer NN
    """
    def __init__(
        self,
        n_actions: int,
        action_range: tf.float32,
        action_min: tf.float32,
        layer_1_dims: int = 512,
        layer_2_dims: int = 512,
        name:str = 'actor',
        checkpoint_dir: str = 'tmp/checkpoints/ddpg'):

        super(ActorNetwork, self).__init__()
        self.layer_1_dims = layer_1_dims
        self.layer_2_dims = layer_2_dims
        self.n_actions = n_actions
        self.action_range = action_range
        self.action_min = action_min

        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            self.model_name + '.h5'
        )

        initializer = keras.initializers.VarianceScaling(
            scale = 0.3333,
            mode = 'fan_in',
            distribution= 'uniform'
        )

        final_initializer = keras.initializers.RandomUniform(minval = -0.003, maxval= 0.003)

        self.layer_1_dense = Dense(
            self.layer_1_dims, 
            activation = 'relu',
            kernel_initializer= initializer)
        self.layer_2_dense = Dense(
            self.layer_2_dims,
            activation = 'relu',
            kernel_initializer= initializer)
        self.mu = Dense(
            self.n_actions, 
            activation = 'tanh', 
            kernel_initializer=final_initializer,
            bias_initializer=final_initializer)

        self.scaling_layer = ScaleLayer(self.action_range, self.action_min)

    def call(self, state: tf.Tensor):
        prob = self.layer_1_dense(state)
        prob = self.layer_2_dense(prob)

        mu_raw = self.mu(prob) 

        mu = self.scaling_layer(mu_raw)

        return mu