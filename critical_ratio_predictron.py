import factory_sim as fact_sim
import numpy as np
import pandas as pd
import math 
# import matplotlib
# import random
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain
import json
import queue

from predictron import Predictron, Replay_buffer

class Config_predictron():
    def __init__(self):
        self.train_dir = './ckpts/predictron_train'
        # self.num_gpus = 1
        
        # adam optimizer:
        self.learning_rate = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.epochs = 5000
        self.batch_size = 128
        self.episode_length = 500
        self.gamma = 0.99
        self.replay_memory_size = 100000
        self.predictron_update_steps = 50
        self.max_depth = 16

sim_time = 5e5
WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)
num_seq_steps = 20

recipes = pd.read_csv('C:/Users/rts/Documents/workspace/WDsim/recipes.csv')
machines = pd.read_csv('C:/Users/rts/Documents/workspace/WDsim/machines.csv')

# with open('ht_seq_mean_w3.json', 'r') as fp:
#     ht_seq_mean_w_l = json.load(fp)
# print(len(machines))

recipes = recipes[recipes.MAXIMUMLS != 0]

# Create the machine dictionary (machine:station)
machine_d = dict()
for index, row in machines.iterrows():
    d = {row[0]:row[1]}
    machine_d.update(d)

# Modifying the above list to match the stations from the two datasets 
a = machines.TOOLSET.unique()
b = recipes.TOOLSET.unique()
common_stations = (set(a) & set(b))
ls = list(common_stations)

# This dictionary has the correct set of stations
modified_machine_dict = {k:v for k,v in machine_d.items() if v in ls}

# Removing uncommon rows from recipes
for index, row in recipes.iterrows():
    if row[2] not in ls:
        recipes.drop(index, inplace=True)

recipes = recipes.dropna()
recipe_dict = dict()
for ht in list(recipes.HT.unique()):
    temp = recipes.loc[recipes['HT'] == ht]
    if len(temp) > 1:
        ls = []
        for index, row in temp.iterrows():
            ls.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]])
        d  = {ht:ls}
        recipe_dict.update(d)
    else:
        ls = []
        ls.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]])
        d = {ht:ls}
        recipe_dict.update(d)

# take only the first num_seq_steps sequence steps for each recipe to reduce the complexity of the simulation.
for ht, step in recipe_dict.items():
    recipe_dict[ht] = step[0:num_seq_steps]

# remove machines which aren't used in the first num_seq_steps for each recipe
used_stations = []
for ht in recipe_dict.keys():
    for step in recipe_dict[ht]:
        used_stations.append(step[0])

used_stations = set(used_stations)

modified_machine_dict = {k:v for k,v in modified_machine_dict.items() if v in list(used_stations)}

# Dictionary where the key is the name of the machine and the value is [station, proc_t]
# machine_dict = {'m0': 's1', 'm2': 's2', 'm1': 's1', 'm3': 's2'}
machine_dict = modified_machine_dict


# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
# recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}
recipes = recipe_dict

wafers_per_box = 4

break_mean = 1e5

repair_mean = 20

n_part_mix = 30

# average lead time for each head type
head_types = recipes.keys()
lead_dict = {}

part_mix = {}


for ht in head_types:
    d = {ht:16000}
    lead_dict.update(d)

    w = {ht:1}
    part_mix.update(w)


####################################################
########## CREATING THE STATE SPACE  ###############
####################################################
def get_state(sim):
    # Calculate the state space representation.
    # This returns a list containing the number of` parts in the factory for each combination of head type and sequence
    # step

    state_rep = sum([sim.n_HT_seq[HT] for HT in sim.recipes.keys()], [])

    # assert state_rep == state_rep2
    # print(len(state_rep))
    # b is a one-hot encoded list indicating which machine the next action will correspond to
    b = np.zeros(len(sim.machines_list))
    b[sim.machines_list.index(sim.next_machine)] = 1
    state_rep.extend(b)
    # Append the due dates list to the state space for making the decision
    rolling_window = [] # This is the rolling window that will be appended to state space
    max_length_of_window = math.ceil(max(sim.lead_dict.values()) / (7*24*60)) # Max length of the window to roll

    current_time = sim.env.now # Calculating the current time
    current_week = math.ceil(current_time / (7*24*60)) #Calculating the current week 

    for key, value in sim.due_wafers.items():
        rolling_window.append(value[current_week:current_week+max_length_of_window]) #Adding only the values from current week up till the window length
        buffer_list = [] # This list stores value of previous unfinished wafers count
        buffer_list.append(sum(value[:current_week]))
        rolling_window.extend([buffer_list])

    c = sum(rolling_window, [])
    state_rep.extend(c) # Appending the rolling window to state space
    # print(len(state_rep))
    return state_rep



####################################################
########## CHOOSING AN ACTION HERE  ################
####################################################

'''
Critical Ratio. The critical ratio (CR) is calculated by dividing the time remaining until
a job’s due date by the total shop time remaining for the job, which is defined as the
setup, processing, move, and expected waiting times of all remaining operations,
including the operation being scheduled. 

CR = (Due date - Today’s date) / (Total shop time remaining)

The difference between the due date and today’s date must be in the same time units as
the total shop time remaining. A ratio less than 1.0 implies that the job is behind schedule, 
and a ratio greater than 1.0 implies that the job is ahead of schedule. The job with
the lowest CR is scheduled next.

'''

def choose_action(sim):
    wafer_list = sim.queue_lists[sim.next_machine.station]

    if len(wafer_list) == 1:
        waf_ = wafer_list[0]
        return waf_

    else:
        cr_ratio = {}
        for waf in wafer_list:
            cr_ = (waf.due_time - sim.env.now) / (sim.get_rem_shop_time(waf.HT, waf.seq, waf.number_wafers))
            cr_ratio[waf] = cr_
        waf_to_choose = min(cr_ratio, key=cr_ratio.get)
        # best_action = (waf_to_choose.HT, waf_to_choose.seq)
        return waf_to_choose

# wt = 'ht_seq_mean_w0.json'
# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, wafers_per_box, part_mix, n_part_mix,
                             break_mean=break_mean, repair_mean=repair_mean)
# start the simulation
my_sim.start()
# Retrieve machine object for first action choice
mach = my_sim.next_machine
# Save the state and allowed actions at the start for later use in training examples
state = get_state(my_sim)
# The action space is a list of tuples of the form [('ht1',0), ('ht1',1), ..., ('ht2', 0), ...] indicating the head
# types and sequence steps for all allowed actions.
action_space = list(chain.from_iterable(my_sim.station_HT_seq.values()))
action_size = len(action_space)
step_counter = 0

# setup of predictron
config = Config_predictron()
config.state_size = len(state)
state_queue = list([])
for i in range(config.episode_length):
    state_queue.append(np.zeros(config.state_size))
reward_queue = list(np.zeros(config.episode_length))
replay_buffer = Replay_buffer(memory_size = config.replay_memory_size)

predictron = Predictron(config)
model = predictron.model
preturn_loss_arr = []
lambda_preturn_loss_arr = []

while my_sim.env.now < sim_time:
    step_counter += 1
    wafer = choose_action(my_sim)
    
    state_episode = state_queue.pop(0)
    state_queue.append(state)
    
    my_sim.run_action(mach, wafer)
    # print('Step Reward:'+ str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward

    reward_queue = [config.gamma*x + reward for x in reward_queue]
    reward_episode = reward_queue.pop(0)
    reward_queue.append(0.)
    
    

    # print(f"state dimension: {len(state)}")
    # print(f"next state dimension: {len(next_state)}")
    # print("action space dimension:", action_size)
    # record the information for use again in the next training example
    mach, allowed_actions = next_mach, next_allowed_actions
    # print("State:", state)
    state = next_state
    
    if (step_counter > config.episode_length):
        replay_buffer.put((state_episode, reward_episode))
        if (step_counter > max(config.batch_size, 2*config.episode_length)) and (step_counter % config.predictron_update_steps) == 0:
            
            data = np.array(replay_buffer.get(config.batch_size))
            states = np.array([np.array(x) for x in data[:,0]])
            states = np.expand_dims(states,-1)
            rewards = np.array([np.array(x) for x in data[:,1]])
            rewards = np.expand_dims(rewards,-1)
            _, preturn_loss, lambda_preturn_loss = model.train_on_batch(states, rewards)
            preturn_loss_arr.append(preturn_loss)
            lambda_preturn_loss_arr.append(lambda_preturn_loss)

    if step_counter % 10000 == 0:
        print(("%.2f" % 100*my_sim.env.now/sim_time)+"% done")
    
for b in range(config.batch_size):
    test_data = np.array([states[b]])
    print(model.predict(test_data)[0],model.predict(test_data)[1], rewards[b])
plt.figure()
plt.plot(preturn_loss_arr)
plt.figure()
plt.plot(lambda_preturn_loss_arr)


# Total wafers produced
# print("Total wafers produced:", len(my_sim.cycle_time))
# # # i = 0
# for ht in my_sim.recipes.keys():
#     # for sequ in range(len(my_sim.recipes[ht])-1):
#     # i += 1
#     # print(len(my_sim.recipes[ht]))
#     # waf = fact_sim.wafer_box(my_sim, 4, ht, my_sim.wafer_index, lead_dict, sequ)
#     # my_sim.wafer_index += 1
#     sequ = len(my_sim.recipes[ht])-1
#     print(ht)
#     print(sequ)
#     print(my_sim.get_rem_shop_time(ht, sequ, 4))

# print(my_sim.get_proc_time('ASGA', 99, 4))
# print(i)
#Wafers of each head type
print("### Wafers of each head type ###")

print(my_sim.lateness)

print(my_sim.complete_wafer_dict)

# ht_seq_mean_w = dict()
# for tup, time_values in my_sim.ht_seq_wait.items():
#     ht_seq_mean_w[tup] = np.mean(time_values)

# with open('ht_seq_mean_wn.json', 'w') as fp:
#     json.dump({str(k): v for k,v in ht_seq_mean_w.items()}, fp)

# Total wafers produced
print("Total wafers produced:", len(my_sim.cycle_time))

# utilization
operational_times = {mach: mach.total_operational_time for mach in my_sim.machines_list}
mach_util = {mach: operational_times[mach]/sim_time for mach in my_sim.machines_list}
mean_util = {station: round(np.mean([mach_util[mach] for mach in my_sim.machines_list if mach.station == station]), 3)
             for station in my_sim.stations}
# stdev_util = {station: np.std(mach_util)

inter_arrival_times = {station: [t_i_plus_1 - t_i for t_i, t_i_plus_1 in zip(my_sim.arrival_times[station],
                                                    my_sim.arrival_times[station][1:])] for station in my_sim.stations}
mean_inter = {station: round(np.mean(inter_ar_ts), 3) for station, inter_ar_ts in inter_arrival_times.items()}
std_inter = {station: round(np.std(inter_ar_ts), 3) for station, inter_ar_ts in inter_arrival_times.items()}
coeff_var = {station: round(std_inter[station]/mean_inter[station], 3) for station in my_sim.stations}

# print(operational_times)
# print(mean_util)
# # print(stdev_util)
# print(inter_arrival_times)
# print(mean_inter)
# print(std_inter)
# print(coeff_var)
#
print(np.mean(my_sim.lateness[-1000:]))

cols = [mean_util, mean_inter, std_inter, coeff_var]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival'])
df = df.transpose()
df.to_csv('util_inter_arr.csv')
# print(df)

# # Plot the time taken to complete each wafer
plt.figure()
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()

