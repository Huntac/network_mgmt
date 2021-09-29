import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(
        self, 
        layer_1_dims: int, 
        layer_2_dims: int,
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

        self.layer_1_dense = Dense(self.layer_1_dims, activation = 'relu')
        self.layer_2_dense = Dense(self.layer_2_dims, activation = 'relu')
        self.q = Dense(1, activation = None)

    def call(self, state, action):
        action_value = self.layer_1_dense(tf.concat([state, action]), axis = 1)
        action_value = self.layer_2_dense(action_value)

        q = self.q(action_value)

        return q
        
class ActorNetwork(keras.Model):
    def __init__(
        self,
        n_actions: int,
        layer_1_dims: int = 512,
        layer_2_dims: int = 512,
        name:str = 'actor',
        checkpoint_dir = 'tmp/checkpoints/ddpg'
    ):
        super(ActorNetwork, self).__init__()
        self.layer_1_dims = layer_1_dims
        self.layer_2_dims = layer_2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            self.model_name + '.h5'
        )

        self.layer_1_dense = Dense(self.layer_1_dims, activation = 'relu')
        self.layer_2_dense = Dense(self.layer_2_dims, activation = 'relu')
        self.mu = Dense(self.n_actions, activation = 'sigmoid')

    def call(self, state):
        prob = self.layer_1_dense(state)
        prob = self.layer_2_dense(prob)

        mu = self.mu(prob)  # output will be adjusted by action bounds in agent class

        return mu