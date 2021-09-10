import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Add, Lambda, Subtract, Average
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from baselines.common.schedules import LinearSchedule

########################################################################################################################################
#################################################################### CREATING Deep Q-learning Class ####################################
########################################################################################################################################

class DQN:
    def __init__(self, state_space_dim, action_space, gamma=0.9, epsilon_decay=0.8, tau=0.125, learning_rate=0.005, epsilon_max=1, batch_size=32, epsilon_min = 0., nstep=1, prioritized_replay_beta_iters=None, seed=None):
        self.state_space_dim = state_space_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.batch_size = batch_size
        self.random_epsilon = random.Random()
        self.random_sample = random.Random()
        if seed is not None:
            self.random_epsilon.seed(seed)
            self.random_sample.seed(seed)
        
        self.nstep = nstep
        self.prioritized_replay_alpha=0.6
        self.prioritized_replay_beta0=0.4
        self.prioritized_replay_beta_iters=prioritized_replay_beta_iters
        self.prioritized_replay_eps=1e-6
        
        self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                       initial_p=self.prioritized_replay_beta0,
                                       final_p=1.0)
        
        self.memory = PrioritizedReplayBuffer(size = 10000, seed=seed, alpha=self.prioritized_replay_alpha)

    # Create the neural network model to train the q function
    def create_model(self):
        # model = Sequential()
        # model.add(layers.Dense(400, input_dim= self.state_space_dim, activation='relu'))
        # model.add(layers.Dense(250, activation='relu'))
        # model.add(layers.Dense(125, activation='relu'))
        # model.add(layers.Dense(len(self.action_space)))
        # model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        
        state_input = Input((self.state_space_dim,))
        backbone_1 = Dense(400, activation='relu')(state_input)
        backbone_2 = Dense(250, activation='relu')(backbone_1)
        backbone_3 = Dense(125, activation='relu')(backbone_2)
        value_output = Dense(1)(backbone_3)
        advantage_output1 = Dense(len(self.action_space))(backbone_3)
        advantage_output1_avg = keras.backend.mean(advantage_output1, axis=-1, keepdims=True)
        advantage_output2 = Subtract()([advantage_output1, advantage_output1_avg])
        output = Add()([value_output, advantage_output2])
        model = tf.keras.Model(state_input, output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def choose_action(self, state, allowed_actions, use_epsilon=True, return_value=False):
        if use_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            r = self.random_epsilon.random()
            if r < self.epsilon:
                # print("******* CHOOSING A RANDOM ACTION *******")
                if return_value:
                    state = np.array(state).reshape(1, self.state_space_dim)
                    pred = self.model.predict(state)
                    pred = sum(pred.tolist(), [])
                    idx = self.random_epsilon.choice(range(len(allowed_actions)))
                    return allowed_actions[idx], pred[self.action_space.index(allowed_actions[idx])]
                else:
                    return self.random_epsilon.choice(allowed_actions)
        # print(state)
        # print(len(state))
        state = np.array(state).reshape(1, self.state_space_dim)
        pred = self.model.predict(state)
        pred = sum(pred.tolist(), [])
        value = []
        for item in allowed_actions:
            value.append(pred[self.action_space.index(item)])
        # print(" ********************* CHOOSING A PREDICTED ACTION **********************")
        if return_value:
            idx = np.argmax(value)
            return allowed_actions[idx], value[idx]
        else:
            return allowed_actions[np.argmax(value)]
    
    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def calculate_value_of_action(self, state, allowed_actions):
        state = np.array(state).reshape(1, self.state_space_dim)
        pred = self.model.predict(state)
        pred = sum(pred.tolist(), [])
        temp = []
        for item in allowed_actions:
            temp.append(pred[self.action_space.index(item)])
        # print(" ********************* CHOOSING A PREDICTED ACTION **********************")
        return np.max(temp)

    # Create replay buffer memory to sample randomly
    def remember(self, state, action, reward, next_state, next_allowed_actions):
        self.memory.add(state, action, reward, next_state, next_allowed_actions, done=False)

    # Build the replay buffer # Train the model
    def replay(self, t, extern_target_model = None):        
        if len(self.memory) < self.batch_size:
            return
        
        if extern_target_model:
            target_model = extern_target_model
        else: 
            target_model = self.target_model
            
        samples = self.memory.sample(self.batch_size, beta=self.beta_schedule.value(t))
        (states, actions, rewards, new_states, new_allowed_actions, dones, weights, batch_idxes) = samples
        states = np.array(states).reshape(self.batch_size, self.state_space_dim)
        preds = self.model.predict(states)
        target_preds = preds.copy()
        action_ids = [self.action_space.index(action) for action in actions]
        # if done:
        #     target[0][action_id] = reward
        # else:
            # take max only from next_allowed_actions
        new_states = np.array(new_states).reshape(self.batch_size, self.state_space_dim)
        if extern_target_model:
            _, next_preds = target_model.predict(new_states) #using lambda predictions
        else:
            next_preds = target_model.predict(new_states)
            next_action_preds = self.model.predict(new_states)
        # next_preds = next_preds.tolist()
        # print("new_allowed_actions:", new_allowed_actions)
        td_errors = []
        for b in range(self.batch_size):
            t = []
            t_act = []
            if extern_target_model:
                next_target = next_preds[b]
            else:
                for it in new_allowed_actions[b]:
                    t.append(next_preds[b][self.action_space.index(it)])
                    t_act.append(next_action_preds[b][self.action_space.index(it)])
                next_target = t[np.argmax(t_act)]
            
            target_preds[b][action_ids[b]] = rewards[b] + (self.gamma**self.nstep) * next_target
            td_errors.append(preds[b][action_ids[b]] - target_preds[b][action_ids[b]])
            
        loss = self.model.train_on_batch(states, target_preds)
        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
        self.memory.update_priorities(batch_idxes, new_priorities)
        
        return loss


    # Update our target network
    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    # Save our model
    def save_model(self, fn):
        self.model.save(fn)
        
    # Load model
    def load_model(self, model_dir):
        self.model.load_weights(model_dir)
        
    
class ReplayBuffer(object):
    def __init__(self, size, seed=None):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.random_sample = random.Random()
        if seed is not None:
            self.random_sample.seed(seed)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, next_allowed_action, done):
        data = (obs_t, action, reward, obs_tp1, next_allowed_action, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, next_allowed_actions_arr, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, next_allowed_actions, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            next_allowed_actions_arr.append(next_allowed_actions)
            dones.append(done)
        return np.array(obses_t), actions, np.array(rewards), np.array(obses_tp1), next_allowed_actions_arr, np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        next_allowed_actions_batch: np.array
            next set of allowed actions for the next obs
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [self.random_sample.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, seed=None, alpha=0.6):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = self.random_sample.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)    