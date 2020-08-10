import factory_sim as fact_sim
import numpy as np
import pandas as pd
import math 
# import matplotlib
# import random
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain
from keras.models import load_model

sim_time = 3e5
WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)
num_seq_steps = 20

recipes = pd.read_csv('C:/Users/rts/Documents/workspace/WDsim/recipes.csv')
machines = pd.read_csv('C:/Users/rts/Documents/workspace/WDsim/machines.csv')
model_dir = "DQN_predictron.h5"

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
    if (row[2] not in ls) or (row[3] == 0 and row[4] == 0):
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

machine_dict.update({'MV3PM3': '602B'})
machine_dict.update({'MV3PM4': '602B'})
machine_dict.update({'MV3PM5': '602B'})
machine_dict.update({'MV3PM6': '602B'})
machine_dict.update({'MV3PM7': '602B'})
machine_dict.update({'MV3PM8': '602B'})
machine_dict.update({'MV3PM9': '602B'})
machine_dict.update({'MV3PM10': '602B'})
machine_dict.update({'MV3PM11': '602B'})
machine_dict.update({'MV3PM12': '602B'})
machine_dict.update({'MV3PM13': '602B'})
machine_dict.update({'MV3PM14': '602B'})
machine_dict.update({'MV3PM15': '602B'})
machine_dict.update({'MV3PM16': '602B'})
machine_dict.update({'MV3PM17': '602B'})
machine_dict.update({'MV3PM18': '602B'})
machine_dict.update({'MV3PM19': '602B'})
machine_dict.update({'MV3PM20': '602B'})
machine_dict.update({'MV3PM21': '602B'})
machine_dict.update({'MV3PM22': '602B'})
machine_dict.update({'DNS-42': 'SCRUBBER'})
machine_dict.update({'DNS-43': 'SCRUBBER'})
machine_dict.update({'DNS-44': 'SCRUBBER'})
machine_dict.update({'DNS-45': 'SCRUBBER'})
machine_dict.update({'DNS-46': 'SCRUBBER'})
machine_dict.update({'FSI015': 'FSI DEV'})
machine_dict.update({'FSI016': 'FSI DEV'})
machine_dict.update({'FSI017': 'FSI DEV'})
machine_dict.update({'FSI018': 'FSI DEV'})
machine_dict.update({'DUV005': 'DUV 193'})
machine_dict.update({'DUV006': 'DUV 193'})
machine_dict.update({'DUV007': 'DUV 193'})
machine_dict.update({'DUV008': 'DUV 193'})
machine_dict.update({'ASHER009': 'ASH IM'})
machine_dict.update({'ASHER0010': 'ASH IM'})
machine_dict.update({'ASHER0011': 'ASH IM'})

# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
# recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}
recipes = recipe_dict

wafers_per_box = 4

break_mean = 1e5

repair_mean = 120

n_part_mix = 30

# average lead time for each head type
head_types = recipes.keys()
lead_dict = {}

part_mix = {}


for ht in head_types:
    d = {ht:1500}
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
    return state_rep




#####################################################################
######################### LOADING THE TRAINED POLICY ################
#####################################################################
model = load_model(model_dir)


# Action function to choose the best action given the q-function if not exploring based on epsilon
def choose_action(state, allowed_actions, action_space):
    state = np.array(state).reshape(1, state_size)
    pred = model.predict(state)
    pred = sum(pred.tolist(), [])
    temp = []
    for item in allowed_actions:
        temp.append(pred[action_space.index(item)])
    print(" ********************* CHOOSING A PREDICTED ACTION **********************")
    return allowed_actions[np.argmax(temp)]




# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, wafers_per_box, part_mix, n_part_mix)
# start the simulation
my_sim.start()
# Retrieve machine object for first action choice
mach = my_sim.next_machine
# Save the state and allowed actions at the start for later use in training examples
state = get_state(my_sim)
allowed_actions = my_sim.allowed_actions
# The action space is a list of tuples of the form [('ht1',0), ('ht1',1), ..., ('ht2', 0), ...] indicating the head
# types and sequence steps for all allowed actions.
action_space = list(chain.from_iterable(my_sim.station_HT_seq.values()))
action_size = len(action_space)
state_size = len(state)


order_count = 0

while my_sim.env.now < sim_time:
    action = choose_action(state, allowed_actions, action_space)

    wafer_choice = next(wafer for wafer in my_sim.queue_lists[mach.station] if wafer.HT == action[0] and wafer.seq ==
                        action[1])

    my_sim.run_action(mach, wafer_choice)
    print('Step Reward:'+ str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward

    print(f"state dimension: {len(state)}")
    print(f"next state dimension: {len(next_state)}")
    print("action space dimension:", action_size)
    # record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    # print("State:", state)


    # Record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state


# Total wafers produced
print("Total wafers produced:", len(my_sim.cycle_time))


#Wafers of each head type
print("### Wafers of each head type ###")
print(my_sim.complete_wafer_dict)
#
# # Plot the time taken to complete each wafer
# plt.plot(my_sim.lateness)
# plt.xlabel("Wafers")
# plt.ylabel("Lateness")
# plt.title("The amount of time each wafer was late (DQN)")
# plt.show()

print(my_sim.lateness)

print(np.mean(my_sim.lateness[-1000:]))

# Plot the time taken to complete each wafer
plt.plot(my_sim.lateness, '.')
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()

# Plot the time taken to complete each wafer
plt.plot(my_sim.cumulative_reward_list)
plt.xlabel("step")
plt.ylabel("Cumulative Reward")
plt.title("The sum of all rewards up until each time step")
plt.show()




