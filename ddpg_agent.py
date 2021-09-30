import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from ddpg_utils import ReplayBuffer
from ddpg_networks import ActorNetwork, CriticNetwork


class Agent:
    """
        DDPG Agent class
    """
    def __init__(
        self,
        input_dims: tuple,
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
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.action_range = self.max_action - self.min_action
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
        # the weights of these networks will be updated via Agent.update_network_parameters
        self.actor_target.compile(optimizer = Adam(learning_rate = alpha))
        self.critic_target.compile(optimizer = Adam(learning_rate = beta))

        # Hard copy initial weights from local networks to target networks
        #   so the networks are identical
        self.update_network_parameters(tau = 1.0)


    def update_network_parameters(self, tau: float= None):
        """
        Apply soft update from local net weights to target net weights
        """
        if tau is None:
            # Allow to hard copy weights when agent is initialized
            tau = self.tau

        weights = []
        targets = self.actor_target.weights
        for i, weight in enumerate(self.actor_local.weights):
            # apply soft updates from local network -> target network
            weights.append(weight * tau + targets[i]*(1-tau))
        self.actor_target.set_weights(weights)

        weights = []
        targets = self.critic_target.weights
        for i, weight in enumerate(self.critic_local.weights):
            # apply soft updates from local network -> target network
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


    def choose_action(self, observation: np.ndarray, evaluate: bool = False):
        """
        Choose an action by passing state to self.actor_local
          
        Scale action according to environment.
        If not evaluating apply noise for exploration.
        Clip Resulting actions if outside of environment bounds.
        """
        state = tf.convert_to_tensor([observation], dtype = tf.float32)
        actions = self.actor_local(state)
        if not evaluate:
            # if not evaluating add noise for exploration
            actions += tf.random.normal(
                shape=[self.n_actions], 
                mean = 0, 
                stddev = self.nosie)

        # scale sigmoid value with action range
        actions = tf.math.multiply(actions, self.action_range)

        # clip scaled action to be within environment action range
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions.numpy()


    def learn(self):
        """
        Adjust Network weights based on stored experiences

        Calculate critic and actor loss and update local network
          weights based on loss grads.
        Apply soft update to target networks.
        """
        if len(self.memory) < self.batch_size:
            # Only learn once enough memories have been stored
            return

        experiences = self.memory.sample()

        # Unpack experience components
        states = np.vstack(e.state for e in experiences if e is not None)
        actions = np.vstack(e.action for e in experiences if e is not None)
        rewards = np.vstack(e.reward for e in experiences if e is not None)
        dones = np.vstack(e.done for e in experiences if e is not None)
        next_states = np.vstack(e.next_state for e in experiences if e is not None)

        # conver experiences to tensor
        states = tf.convert_to_tensor(states, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
        actions = tf.convert_to_tensor(actions, dtype = tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)

        # Train 
        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states)
            next_values = tf.squeeze(self.critic_target(next_states, next_actions), 1)
            values = tf.squeeze(self.critic_local(states, actions), 1)

            # Target Q value is the observed reward plus the discounted reward of the 
            #   following time step unless the reward is from a terminal step
            target = reward + self.gamma*next_values*(1-dones)
            critic_loss = keras.losses.MSE(target, values)

        critic_network_gradient = tape.gradient(
            critic_loss, 
            self.critic_local.trainable_variables)
        
        self.critic_local.optimizer.apply_gradients(
            zip(critic_network_gradient, self.critic_local.trainable_variables)
        )

        # Update Actor weights based on critic_local assessment of actor_local policy. 
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor_local(states)
            # Taking the negative critic_local Q value to do gradient ascent
            actor_loss = -self.critic_local(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        # Because we're passing actor_local predictions into critic_local as input 
        #   tf.GradientTape() allows us to calculate the gradient of the negative 
        #   Q value from critic_local w.r.t actor_local's trainable variables
        actor_network_gradient = tape.gradient(actor_loss,
        self.actor_local.trainable_variables)

        self.actor_local.optimizer.apply_gradient(
            zip(actor_network_gradient, self.actor_local.trainable_variables)
        )

        self.update_network_parameters()
        