from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


########################################################################################################################################
#################################################################### CREATING Deep Q-learning Class ####################################
########################################################################################################################################

class DQN:
    def __init__(self, state_space_dim, action_space, gamma=0.9, epsilon_decay=0.8, tau=0.125, learning_rate=0.005, epsilon_max = 1):
        self.state_space_dim = state_space_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = 0.0
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.batch_size = 32

    # Create the neural network model to train the q function
    def create_model(self):
        model = Sequential()
        model.add(Dense(400, input_dim= self.state_space_dim, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(125, activation='relu'))
        model.add(Dense(len(self.action_space)))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def choose_action(self, state, allowed_actions):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        r = np.random.random()
        if r < self.epsilon:
            # print("******* CHOOSING A RANDOM ACTION *******")
            return random.choice(allowed_actions)
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
            
        samples = random.sample(self.memory, self.batch_size)
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
        # next_preds = next_preds.tolist()
        t = []
        # print("new_allowed_actions:", new_allowed_actions)
        for b in range(self.batch_size):
            if extern_target_model:
                next_target = next_preds[b]
            else:
                for it in new_allowed_actions[b]:
                    t.append(next_preds[b][self.action_space.index(it)])
                next_target = max(t)
            
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