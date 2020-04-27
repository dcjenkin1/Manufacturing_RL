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
import PG_Class

sim_time = 5e5
WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)
num_seq_steps = 20

recipes = pd.read_csv('~/Documents/workspace/WDsim/recipes.csv')
machines = pd.read_csv('~/Documents/workspace/WDsim/machines.csv')

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

# Removing unncommon rows from recipes 
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

repair_mean = 120

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


# Simple pad utility function
def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


####################################################
########## CREATING THE STATE SPACE  ###############
####################################################
def get_state(sim):
    # Calculate the state space representation.
    # This returns a list containing the number of` parts in the factory for each combination of head type and sequence
    # step

    state_rep = sum([sim.n_HT_seq[HT] for HT in sim.recipes.keys()], [])

    # assert state_rep == state_rep2
    print(len(state_rep))
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
    print(len(state_rep))
    return state_rep


# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, wafers_per_box, part_mix, n_part_mix, break_mean, repair_mean)
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

# create the pol_grad object with the appropriate length of state and action space
pol_grad = PG_Class.PolGrad(action_space, len(state))

# def my_custom_loss():
#     def custom_loss(y_pred, y_true, discounted_episode_rewards):
#         neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
#         loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards)
#         return loss
#
#
# pol_grad = load_model("PG_model.h5", custom_objects={'custom_loss': my_custom_loss})

episode_states, episode_actions, allRewards, episode_allowed_a = [],[],[],[]


while my_sim.env.now < sim_time:
    episode_states.append(state)
    episode_allowed_a.append(allowed_actions)
    print("State shape is :", len(state))
    action = pol_grad.choose_action(state, allowed_actions)
    action_ = np.zeros(action_size)
    action_[action] = 1
    episode_actions.append(action_)
    
    action = action_space[action]

    wafer_choice = next(wafer for wafer in my_sim.queue_lists[mach.station] if wafer.HT == action[0] and wafer.seq ==
                        action[1])
    
    if my_sim.order_completed:
        # Calculate discounted reward
        episode_rewards_ = np.ones(np.asarray(episode_states).shape[0])
        episode_rewards_ *= my_sim.step_reward
        pol_grad.train_policy_gradient(np.asarray(episode_states), np.asarray(episode_actions), episode_rewards_, episode_allowed_a)
        
        # Reset the transition stores
        episode_states, episode_actions, episode_allowed_a = [],[],[]

    my_sim.run_action(mach, wafer_choice)
    state = get_state(my_sim)
    allowed_actions = my_sim.allowed_actions
    mach = my_sim.next_machine

    print(my_sim.order_completed)
    print(state)
    print(my_sim.step_reward)


# Save the trained PG policy network
pol_grad.save_model("PG_model.h5")




#Wafers of each head type
print("### Wafers of each head type ###")
print(my_sim.complete_wafer_dict)

# Total wafers produced
print("Total wafers produced:", len(my_sim.cycle_time))
print(np.mean(my_sim.lateness[-1000:]))

# Plot the time taken to complete each wafer
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The time each wafer was late")
plt.show()

# Plot the time taken to complete each wafer
plt.plot(my_sim.cumulative_reward_list)
plt.xlabel("step")
plt.ylabel("Cumulative Reward")
plt.title("The sum of all rewards up until each time step")
plt.show()






