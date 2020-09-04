# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU') # Use this to run on CPU only
import factory_sim as fact_sim
import numpy as np
import pandas as pd
import math 
import os
# import matplotlib
# import random
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain
from keras.models import load_model
import argparse
import datetime
import json


parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--state_rep_size", default='32', help="Size of the state representation")
parser.add_argument("--sim_time", default=1e5, type=int, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='./b20_setup/', help="Path to factory setup files")
parser.add_argument("--model_dir", default='data/models/DQN_model_5e5.h5', help="Path to DQN model")
parser.add_argument("--seed", default=2, type=int, help="seed for random functions")
args = parser.parse_args()


seed = args.seed

id = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())

sim_time = args.sim_time
model_dir = args.model_dir


WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)

with open(args.factory_file_dir+'break_repair_wip.json', 'r') as fp:
    break_repair_WIP = json.load(fp)

with open(args.factory_file_dir+'machines.json', 'r') as fp:
    machine_dict = json.load(fp)

with open(args.factory_file_dir+'recipes.json', 'r') as fp:
    recipes = json.load(fp)

with open(args.factory_file_dir+'due_date_lead.json', 'r') as fp:
    lead_dict = json.load(fp)

with open(args.factory_file_dir+'part_mix.json', 'r') as fp:
    part_mix = json.load(fp)
    
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
    # print(" ********************* CHOOSING A PREDICTED ACTION **********************")
    return allowed_actions[np.argmax(temp)]




# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, part_mix, break_repair_WIP['n_batch_wip'],
                             break_mean=break_repair_WIP['break_mean'], repair_mean=break_repair_WIP['repair_mean'], seed=seed)
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
step_counter = 0
while my_sim.env.now < sim_time:
    action = choose_action(state, allowed_actions, action_space)

    wafer_choice = next(wafer for wafer in my_sim.queue_lists[mach.station] if wafer.HT == action[0] and wafer.seq ==
                        action[1])

    my_sim.run_action(mach, wafer_choice)
    # print('Step Reward:'+ str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward

    # print(f"state dimension: {len(state)}")
    # print(f"next state dimension: {len(next_state)}")
    # print("action space dimension:", action_size)
    # record the information for use again in the next training example
    # mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    # print("State:", state)
    

    # Record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    
    step_counter += 1
    if step_counter % 1000 == 0 and step_counter > 1:
        print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
        print("Mean lateness: ", np.mean(my_sim.lateness))


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
print(np.mean(my_sim.lateness))
print(np.mean(my_sim.lateness[-10000:]))

# utilization
operational_times = {mach: mach.total_operational_time for mach in my_sim.machines_list}
mach_util = {mach: operational_times[mach]/sim_time for mach in my_sim.machines_list}
mean_util = {station: round(np.mean([mach_util[mach] for mach in my_sim.machines_list if mach.station == station]), 3)
             for station in my_sim.stations}
# mean_mach_takt_times = {mach: np.mean(mach.takt_times) for mach in my_sim.machines_list}
# std_mach_takt_times = {mach: round(np.std(mach.takt_times), 3) for mach in my_sim.machines_list}
#
# mean_station_takt_times = {station: round(np.mean([mean_mach_takt_times[mach] for mach in my_sim.machines_list if
#                                          mach.station == station and not np.isnan(mean_mach_takt_times[mach])]), 3) for
#                            station in my_sim.stations}
# mean_station_takt_times = {station: round(1/sum([1/mean_mach_takt_times[mach] for mach in my_sim.machines_list if
#                                          mach.station == station]), 3) for station in my_sim.stations}

parts_per_station = {station: sum([mach.parts_made for mach in my_sim.machines_list if mach.station == station]) for
                     station in my_sim.stations}

station_wait_times = {station: np.mean(sum([my_sim.ht_seq_wait[(ht, seq)] for ht, seq in my_sim.station_HT_seq[station]], [])) for
                      station in my_sim.stations}

# stdev_util = {station: np.std(mach_util)

inter_arrival_times = {station: [t_i_plus_1 - t_i for t_i, t_i_plus_1 in zip(my_sim.arrival_times[station],
                                                    my_sim.arrival_times[station][1:])] for station in my_sim.stations}
mean_inter = {station: round(np.mean(inter_ar_ts), 3) for station, inter_ar_ts in inter_arrival_times.items()}
std_inter = {station: round(np.std(inter_ar_ts), 3) for station, inter_ar_ts in inter_arrival_times.items()}
coeff_var = {station: round(std_inter[station]/mean_inter[station], 3) for station in my_sim.stations}
machines_per_station = {station: len([mach for mach in my_sim.machines_list if mach.station == station]) for station in
                        my_sim.stations}


path,file=os.path.split(args.model_dir)
data_dir = './'+path+'/data/seed_'+str(seed)+'/'+str(id)+'/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

np.savetxt(data_dir+'wafer_lateness.csv', np.array(my_sim.lateness), delimiter=',')
cols = [mean_util, mean_inter, std_inter, coeff_var, machines_per_station, station_wait_times]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival', 'machines_per_station', 'mean_wait_time'])
df = df.transpose()
df.to_csv(data_dir+'util.csv')

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



