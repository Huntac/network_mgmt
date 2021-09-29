import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from ddpg_utils import ReplayBuffer
from ddpg_networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        input_dims,
        n_actions: int,
        alpha: float = 0.001,
        beta: float = 0.002,
        env: gym.env = None,
        gamma: float = 0.99,
        tau: float = 0.001,
        noise: float = 0.1,
        layer_1_dims: int = 400,
        layer_2_dims: int = 300,
        buffer_size: int = 1000000,
        batch_size: int = 64
    ):
        # initializes utilities
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # DDPG parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.n_actions = n_actions
        self.nosie = noise

        # Networks
        self.actor_local = ActorNetwork(n_actions = n_actions, name = 'actor_local')
        self.actor_target = ActorNetwork(n_actions = n_actions, name = 'actor_target')
        
        self.critic_local = CriticNetwork(name = 'critic_local')
        self.critic_target = CriticNetwork(name = 'critic_target')

        self.actor_local.compile(optimizer = Adam(learning_rate = alpha))
        self.critic_local.compile(optimizer = Adam(learning_rate = beta))

        # optimizer only passed so that these models compile
        # the weights of these networks will be updated via Agent.soft_update
        self.actor_target.compile(optimizer = Adam(learning_rate = alpha))
        self.critic_target.compile(optimizer = Adam(learning_rate = beta))

        self.update_network_parameters(tau  = 1)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.actor_target.weights
        for i, weight in enumerate(self.actor_local.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.actor_target.set_weights(weights)

        weights = []
        targets = self.critic_target.weights
        for i, weight in enumerate(self.critic_local.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.critic_target.set_weights(weights)

    def save_models(self):
        print('saving models')
        self.actor_local.save_weights(self.actor_local.checkpoint_file)
        self.actor_target.save_weights(self.actor_target.checkpoint_file)
        self.critic_local.save_weights(self.critic_local.checkpoint_file)
        self.critic_target.save_weights(self.critic_target.checkpoint_file)

    def load_models(self):
        print('loading models')
        self.actor_local.load_weights(self.actor_local.checkpoint_file)
        self.actor_target.load_weights(self.actor_target.checkpoint_file)
        self.critic_local.load_weights(self.critic_local.checkpoint_file)
        self.critic_target.load_weights(self.critic_target.checkpoint_file)

    def choose_action(self, observation, evalute = False):
        state = tf.convert_to_tensor([observation], dtype = tf.float32)
        actions = self.actor_local(state)
        if not evaluate:
            actions += tf.random.normal(
                shape=[self.n_actions], 
                mean = 0, 
                stddev = self.nosie)

        # still need to scale sigmoid by min_action and max_action

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions.numpy()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample()

        states = np.vstack(e.state for e in experiences if e is not None)
        actions = np.vstack(e.action for e in experiences if e is not None)
        rewards = np.vstack(e.reward for e in experiences if e is not None)
        dones = np.vstack(e.done for e in experiences if e is not None)
        next_states = np.vstack(e.next_state for e in experiences if e is not None)

        states = tf.convert_to_tensor(states, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
        actions = tf.convert_to_tensor(actions, dtype = tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)

        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states)
            next_values = tf.squeeze(self.critic_target(next_states, next_actions), 1)
            values = tf.squeeze(self.critic_local(states, actions), 1)
            target = reward + self.gamma*next_values*(1-dones)
            critic_loss = keras.losses.MSE(target, values)

        critic_network_gradient = tape.gradient(
            critic_loss, 
            self.critic_local.trainable_variables)
        
        self.critic_local.optimizer.apply_gradients(
            zip(critic_network_gradient, self.critic_local.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor_local(states)
            actor_loss = -self.critic_local(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
        self.actor_local.trainable_variables)

        self.actor_local.optimizer.apply_gradient(
            zip(actor_network_gradient, self.actor_local.trainable_variables)
        )
        