import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Add, Lambda, Subtract, Average
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random


########################################################################################################################################
#################################################################### CREATING Deep Q-learning Class ####################################
########################################################################################################################################

class DQN:
    def __init__(self, state_space_dim, action_space, gamma=0.9, epsilon_decay=0.8, tau=0.125, learning_rate=0.005, epsilon_max=1, batch_size=32, epsilon_min = 0., seed=None):
        self.state_space_dim = state_space_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.batch_size = batch_size
        self.random_epsilon = random.Random()
        self.random_sample = random.Random()
        if seed is not None:
            self.random_epsilon.seed(seed)
            self.random_sample.seed(seed)

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
    def choose_action(self, state, allowed_actions, use_epsilon=True):
        if use_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            r = self.random_epsilon.random()
            if r < self.epsilon:
                # print("******* CHOOSING A RANDOM ACTION *******")
                return self.random_epsilon.choice(allowed_actions)
        # print(state)
        # print(len(state))
        state = np.array(state).reshape(1, self.state_space_dim)
        pred = self.model.predict(state)
        pred = sum(pred.tolist(), [])
        temp = []
        for item in allowed_actions:
            temp.append(pred[self.action_space.index(item)])
        # print(" ********************* CHOOSING A PREDICTED ACTION **********************")
        return allowed_actions[np.argmax(temp)]
    
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
        self.memory.append([state, action, reward, next_state, next_allowed_actions])

    # Build the replay buffer # Train the model
    def replay(self, extern_target_model = None):        
        if len(self.memory) < self.batch_size:
            return
        
        if extern_target_model:
            target_model = extern_target_model
        else: 
            target_model = self.target_model
            
        samples = self.random_sample.sample(self.memory, self.batch_size)
        states, actions, rewards, new_states, new_allowed_actions = zip(*samples)
        states = np.array(states).reshape(self.batch_size, self.state_space_dim)
        preds = self.model.predict(states)
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
            next_action_preds = model.predict(new_states)
        # next_preds = next_preds.tolist()
        # print("new_allowed_actions:", new_allowed_actions)
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
            
            preds[b][action_ids[b]] = rewards[b] + self.gamma * next_target
            
        return self.model.train_on_batch(states, preds)


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