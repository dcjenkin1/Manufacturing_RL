import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Use this to run on CPU only
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
import argparse
import datetime
import os

from PDN import PDN, Replay_buffer

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
# parser.add_argument("--predictron_model_dir", default='./Predictron_DQN_3e5_dense_32_base.h5', help="Path to the Predictron model")
parser.add_argument("--state_rep_size", default='128', help="Size of the state representation")
parser.add_argument("--sim_time", default=5e5, type=int, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='./b20_setup/', help="Path to factory setup files")
parser.add_argument("--save_dir", default='./pdqn/', help="Path save log files in")
parser.add_argument("--seed", default=0, help="random seed")
args = parser.parse_args()

id = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())

sim_time = args.sim_time

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


# recipes = pd.read_csv(args.factory_file_dir + 'recipes.csv')
# machines = pd.read_csv(args.factory_file_dir + 'machines.csv')
# # predictron_model_dir = args.predictron_model_dir

model_dir = args.save_dir+'models/PDN/srs_'+str(args.state_rep_size)+'/'+str(id)+'/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)
# num_seq_steps = 40



class Config_predictron():
    def __init__(self):
        # self.num_gpus = 1
        
        # adam optimizer:
        self.learning_rate = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon_adam = 1e-8
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        self.l2_weight=0.01
        self.dropout_rate=0.
        
        self.batch_size = 32
        self.episode_length = 50
        self.burnin = 0*3e3
        self.gamma = 0.99
        self.replay_memory_size = 10000
        self.predictron_update_steps = 1
        self.max_depth = 16
        self.state_size = None # set this before running the predictron
        self.action_size = None # set this before running the predictron
        
        self.state_rep_size = args.state_rep_size
        
        self.seed = args.seed


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

# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, part_mix, break_repair_WIP['n_batch_wip'],
                             break_mean=break_repair_WIP['break_mean'], repair_mean=break_repair_WIP['repair_mean'], seed=args.seed)
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


# setup of predictron
config = Config_predictron()
config.state_size = state_size
config.action_size = action_size
config.action_space = action_space
state_queue = list([])
discount_array = []
for i in range(config.episode_length):
    state_queue.append(np.zeros(config.state_size))
    discount_array.append(config.gamma**i)
discount_array=np.array(discount_array)
action_queue = list(np.zeros(config.episode_length))
reward_queue = list(np.zeros(config.episode_length))
replay_buffer = Replay_buffer(memory_size = config.replay_memory_size, seed=config.seed)

pdn = PDN(config)
model = pdn.model
preturn_loss_arr = []
max_preturn_loss = 0
lambda_preturn_loss_arr = []
max_lambda_preturn_loss = 0

predictron_lambda_arr = []
reward_episode_arr = []

step_counter = 0
while my_sim.env.now < sim_time:
    # print(my_sim.env.now)
    action_id = pdn.choose_action(state, allowed_actions)
    
    action = allowed_actions[allowed_actions.index(pdn.action_space[action_id])]

    wafer_choice = next(wafer for wafer in my_sim.queue_lists[mach.station] if wafer.HT == action[0] and wafer.seq == action[1])
    
    state_episode = state_queue.pop(0)
    state_queue.append(state)
    
    action_episode = action_queue.pop(0)
    action_queue.append(action_id)
    
    my_sim.run_action(mach, wafer_choice)
    # print('Step Reward:'+ str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward
    
    reward_episode = np.sum(np.array(reward_queue)*discount_array)
    reward_queue.pop(0)
    reward_queue.append(reward)
    
    if my_sim.env.now > config.burnin:
        step_counter += 1
        
    if step_counter > config.episode_length:
        replay_buffer.put((state_episode, action_episode, reward_episode))
        if step_counter > config.episode_length+config.batch_size and (step_counter % config.predictron_update_steps) == 0:
            
            data = np.array(replay_buffer.get(config.batch_size))
            states = np.array([np.array(x) for x in data[:,0]])
            states = np.expand_dims(states,-1)
            actions = np.array([np.array(x) for x in data[:,1]], dtype=int)
            rewards = np.array([np.array(x) for x in data[:,2]])
            # rewards = np.expand_dims(rewards,-1)
            
            
            preds = model.predict(states)
            preturn_pred = []
            for idx, p in enumerate(preds[0]):
                p[:,actions[idx]] = rewards[idx]
                preturn_pred.append(p)
            preds[0] = np.array(preturn_pred)
            preds[1][:,:,actions] = rewards
            targets = preds
            _, preturn_loss, lambda_preturn_loss = model.train_on_batch(states, targets)
            model.train_on_batch(states) # consistency update
            # max_lambda_preturn_loss = max(max_lambda_preturn_loss, lambda_preturn_loss)
            # max_preturn_loss = max(max_preturn_loss, preturn_loss)
            preturn_loss_arr.append(preturn_loss)
            lambda_preturn_loss_arr.append(lambda_preturn_loss)
            
    if step_counter % 1000 == 0 and step_counter > 1:
        print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
        
        if step_counter > config.episode_length+config.batch_size:
            print("running mean % of max preturn loss: ", "%.2f" % (100*np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):])/max_preturn_loss), "\t\t", np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):]))
            print("running mean % of max lambda preturn loss: ", "%.2f" % (100*np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):])/max_lambda_preturn_loss), "\t\t", np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):]))
            pdn_result = model.predict([state])
            predictron_lambda_arr.append(np.max(pdn_result[1],axis=-1))
            reward_episode_arr.append(reward_episode)

            print(pdn_result[0][:,:,np.argmax(pdn_result[1][0,0],axis=-1)][0],predictron_lambda_arr[-1][0], reward_episode)
        
    # print(f"state dimension: {len(state)}")
    # print(f"next state dimension: {len(next_state)}")
    # print("action space dimension:", action_size)
    # record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    # print("State:", state)
    
# Save the trained Predictron network
model.save(args.save_dir+'PDN_' + id + 'seed' + str(args.seed) + '.h5')


plt.figure()
plt.plot(preturn_loss_arr)
plt.figure()
plt.plot(lambda_preturn_loss_arr)
plt.figure()
plt.plot(np.array(predictron_lambda_arr).flatten(), label='Predictron')
plt.plot(reward_episode_arr, label='GT')
plt.title("Value estimate")
plt.legend()

predictron_error = np.abs(np.array(predictron_lambda_arr).flatten()-np.array(reward_episode_arr))
predictron_error_avg = [predictron_error[0]]
alpha = 0.05
for i in range(len(predictron_error)-1):
    predictron_error_avg.append(predictron_error_avg[i]*(1-alpha) + predictron_error[i+1]*alpha)
plt.figure()
plt.plot(predictron_error, label='Predictron')
plt.plot(predictron_error_avg, label='Running average')
plt.title("Absolute value estimate error")
plt.legend()




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
# print(operational_times)
# print(mean_util)
# # print(stdev_util)
# print(inter_arrival_times)
# print(mean_inter)
# print(std_inter)
# print(coeff_var)
#
print(np.mean(my_sim.lateness[-10000:]))

cols = [mean_util, mean_inter, std_inter, coeff_var]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival'])
df = df.transpose()
df.to_csv(args.save_dir+'PDN_util_'+str(id)+'_seed_'+str(args.seed)+'.csv')

np.savetxt(args.save_dir+'PDN_wafer_lateness_'+str(id)+'_seed_'+str(args.seed)+'.csv', np.array(my_sim.lateness), delimiter=',')
# print(df)

# # Plot the time taken to complete each wafer
plt.figure()
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()

