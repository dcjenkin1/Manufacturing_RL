import factory_sim as fact_sim
import numpy as np
import pandas as pd
import math 
# import matplotlib
# import random
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain
import argparse
import datetime
import json
import queue

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
# parser.add_argument("--predictron_model_dir", default='./Predictron_DQN_3e5_dense_32_base.h5', help="Path to the Predictron model")
parser.add_argument("--state_rep_size", default='32', help="Size of the state representation")
parser.add_argument("--sim_time", default=5e5, type=int, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='./b20_setup/', help="Path to factory setup files")
parser.add_argument("--save_dir", default='./pdqn/', help="Path save log files in")
args = parser.parse_args()
s = args.save_dir

sim_time = args.sim_time

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
            cr_ = (waf.due_time - sim.env.now) / (sim.get_rem_shop_time(waf.HT, waf.seq))
            cr_ratio[waf] = cr_
        waf_to_choose = min(cr_ratio, key=cr_ratio.get)
        # best_action = (waf_to_choose.HT, waf_to_choose.seq)
        return waf_to_choose

# wt = 'ht_seq_mean_w0.json'
# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, part_mix, break_repair_WIP['n_batch_wip'],
                             break_mean=break_repair_WIP['break_mean'], repair_mean=break_repair_WIP['repair_mean'])
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

# Save the trained Predictron network
model.save("Predictron_CR_1e5.h5")

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

# print(my_sim.lateness)

# print(my_sim.complete_wafer_dict)

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

print(np.mean(my_sim.lateness[-1000:]))

# cols = [mean_util, mean_inter, std_inter, coeff_var, machines_per_station, station_wait_times]
# df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
#                   'coefficient_of_var_interarrival', 'machines_per_station', 'mean_wait_time'])
# df = df.transpose()
# df.to_csv(args.save_dir+'util'+id+'.csv')

# print(df)

# # Plot the time taken to complete each wafer
plt.figure()
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()
#
# Plot the time taken to complete each wafer
plt.plot(my_sim.cumulative_reward_list)
plt.xlabel("step")
plt.ylabel("Cumulative Reward")
plt.title("The sum of all rewards up until each time step")
plt.show()

