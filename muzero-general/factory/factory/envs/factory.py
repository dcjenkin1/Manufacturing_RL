import gym
from gym import error, spaces, utils
from gym.utils import seeding

import json
import numpy as np

import math
from itertools import chain
from .factory_sim import FactorySim

class FactoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.sim_time = 1e5
        self.factory_file_dir = 'b20_setup/'
        self.seed_ = None
        
        WEEK = 24*7
        self.NO_OF_WEEKS = math.ceil(self.sim_time/WEEK)
        with open(self.factory_file_dir+'break_repair_wip.json', 'r') as fp:
            self.break_repair_WIP = json.load(fp)
      
        with open(self.factory_file_dir+'machines.json', 'r') as fp:
            self.machine_dict = json.load(fp)
      
        with open(self.factory_file_dir+'recipes.json', 'r') as fp:
            self.recipes = json.load(fp)
      
        with open(self.factory_file_dir+'due_date_lead.json', 'r') as fp:
            self.lead_dict = json.load(fp)
        
        with open(self.factory_file_dir+'part_mix.json', 'r') as fp:
            self.part_mix = json.load(fp)
      
        # Create the factory simulation object
        self.my_sim = FactorySim(self.sim_time, self.machine_dict, self.recipes, self.lead_dict, self.part_mix, self.break_repair_WIP['n_batch_wip'],
                                     break_mean=self.break_repair_WIP['break_mean'], repair_mean=self.break_repair_WIP['repair_mean'], seed=self.seed_)
        # start the simulation
        self.my_sim.start()
        state = self.get_state()
        HEIGHT = len(state)
        WIDTH = 1
        N_CHANNELS = 1
        
        self.all_actions = list(chain.from_iterable(self.my_sim.station_HT_seq.values()))

        self.action_space = spaces.Discrete(len(self.all_actions))
        self.observation_space = spaces.Box(low=0, high=256, 
                                    shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    def step(self, action):
        # allowed_actions_idx = [action_space.index(allowed_action) for allowed_action in allowed_actions]
        # action_idx = action_space.index(action)
        
        mach = self.my_sim.next_machine
        mach_action = self.all_actions[action]
        wafer_choice = next(wafer for wafer in self.my_sim.queue_lists[mach.station] if wafer.HT == mach_action[0] 
                            and wafer.seq == mach_action[1])
    
        self.my_sim.run_action(mach, wafer_choice)
        
        next_state = self.get_state()
        
        reward = self.my_sim.step_reward
        
        done =  self.my_sim.env.now >= self.sim_time
        
        info = {}
        
        return next_state, reward, done, info
    
    def reset(self):
        print("Resetting environment")
        # Create the factory simulation object
        self.my_sim = FactorySim(self.sim_time, self.machine_dict, self.recipes, self.lead_dict, self.part_mix, self.break_repair_WIP['n_batch_wip'],
                                     break_mean=self.break_repair_WIP['break_mean'], repair_mean=self.break_repair_WIP['repair_mean'], seed=self.seed_)
        # start the simulation
        self.my_sim.start()
        # Retrieve machine object for first action choice
        self.mach = self.my_sim.next_machine
        state = self.get_state()
        
        
        return state
        
    def render(self, mode='human', close=False):
        pass
    
    def seed(self,seed):
        self.seed_=seed
        # print("Seed set to: ", seed)
        
        state = self.reset() # Resetting env with new seed. Returning state
        return state
    
    def get_state(self):
        # Calculate the state space representation.
        # This returns a list containing the number of` parts in the factory for each combination of head type and sequence
        # step
        state_rep = sum([self.my_sim.n_HT_seq[HT] for HT in self.my_sim.recipes.keys()], [])
    
        # print(len(state_rep))
        # b is a one-hot encoded list indicating which machine the next action will correspond to
        b = np.zeros(len(self.my_sim.machines_list))
        b[self.my_sim.machines_list.index(self.my_sim.next_machine)] = 1
        state_rep.extend(b)
        # Append the due dates list to the state space for making the decision
        rolling_window = [] # This is the rolling window that will be appended to state space
        max_length_of_window = math.ceil(max(self.my_sim.lead_dict.values()) / (7*24*60)) # Max length of the window to roll
    
        current_time = self.my_sim.env.now # Calculating the current time
        current_week = math.ceil(current_time / (7*24*60)) #Calculating the current week 
    
        for key, value in self.my_sim.due_wafers.items():
            rolling_window.append(value[current_week:current_week+max_length_of_window]) #Adding only the values from current week up till the window length
            buffer_list = [] # This list stores value of previous unfinished wafers count
            buffer_list.append(sum(value[:current_week]))
            rolling_window.extend([buffer_list])
    
        c = sum(rolling_window, [])
        state_rep.extend(c) # Appending the rolling window to state space
        # state_rep = np.reshape(state_rep, (1,1,355))
        return state_rep
    
    def legal_actions(self):
        allowed_actions = [self.all_actions.index(allowed_action) for allowed_action in self.my_sim.allowed_actions]
        return allowed_actions
    
    def close(self):        
        self = None