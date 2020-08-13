# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU') # Use this to run on CPU only
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
import DeepQNet
import argparse
import datetime

from predictron import Predictron, Replay_buffer

id = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--dqn_model_dir", default='./DQN_model_5e5.h5', help="Path to the DQN model")
parser.add_argument("--predictron_model_dir", default='./', help="Path to the Predictron model")
parser.add_argument("--state_rep_size", default='32', help="Size of the state representation")
parser.add_argument("--sim_time", default=3e5, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='~/mypath/', help="Path to factory setup files")
parser.add_argument("--save_dir", default='./', help="Path save log files in")
args = parser.parse_args()

sim_time = args.sim_time
recipes = pd.read_csv(args.factory_file_dir + 'recipes.csv')
machines = pd.read_csv(args.factory_file_dir + 'machines.csv')
dqn_model_dir = args.dqn_model_dir
predictron_model_dir = args.predictron_model_dir+'Predictron_DQN_100000.0_full_'+str(args.state_rep_size)+'.h5'

WEEK = 24*7
NO_OF_WEEKS = math.ceil(sim_time/WEEK)
num_seq_steps = 20

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
        self.burnin = 1e4
        self.gamma = 0.99
        self.replay_memory_size = 100000
        self.predictron_update_steps = 50
        self.max_depth = 16
        
        self.DQN_train_steps = 10000
        self.Predictron_train_steps = 5000
        
        self.state_rep_size = args.state_rep_size


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
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, wafers_per_box, part_mix, n_part_mix,
                             break_mean=break_mean, repair_mean=repair_mean)
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
step_counter = 0

# setup of predictron
config = Config_predictron()
config.state_size = state_size
state_queue = list([])
for i in range(config.episode_length):
    state_queue.append(np.zeros(config.state_size))
reward_queue = list(np.zeros(config.episode_length))
replay_buffer = Replay_buffer(memory_size = config.replay_memory_size)

predictron = Predictron(config)
model = predictron.model
model.load_weights(predictron_model_dir)
preturn_loss_arr = []
max_preturn_loss = 0
lambda_preturn_loss_arr = []
max_lambda_preturn_loss = 0

DQN_arr =  []
predictron_lambda_arr = []
reward_episode_arr = []

# Creating the DQN agent
dqn_agent = DeepQNet.DQN(state_space_dim= state_size, action_space= action_space, epsilon_max=0., gamma=0.99)
dqn_agent.load_model(dqn_model_dir)
order_count = 0

TRAIN_DQN = True
step_counter = 0

dqn_loss_arr = []
pred_loss_arr = []
while my_sim.env.now < sim_time:
    
    action = dqn_agent.choose_action(state, allowed_actions)

    wafer_choice = next(wafer for wafer in my_sim.queue_lists[mach.station] if wafer.HT == action[0] and wafer.seq ==
                        action[1])

    my_sim.run_action(mach, wafer_choice)
    # print('Step Reward:' + str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward
    
    if TRAIN_DQN: #DQN
        # Save the example for later training
        dqn_agent.remember(state, action, reward, next_state, next_allowed_actions)
        if step_counter % 1000 == 0 and step_counter > 1:
            print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
        if my_sim.order_completed:
            # After each wafer completed, train the policy network 
            loss = dqn_agent.replay(extern_target_model = predictron.model)
            dqn_loss_arr.append(np.mean(loss))
            order_count += 1
            if order_count >= 1:
                # After every 20 processes update the target network and reset the order count
                dqn_agent.train_target()
                order_count = 0
                # Record the information for use again in the next training example
                
        mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    
    else:  #Predictron
                
        state_episode = state_queue.pop(0)
        state_queue.append(state)
                
        reward_queue = [config.gamma*x + reward for x in reward_queue]
        reward_episode = reward_queue.pop(0)
        reward_queue.append(0.)
            
        if step_counter > config.episode_length:
            replay_buffer.put((state_episode, reward_episode))
            if step_counter > config.episode_length+config.batch_size and (step_counter % config.predictron_update_steps) == 0:
                
                data = np.array(replay_buffer.get(config.batch_size))
                states = np.array([np.array(x) for x in data[:,0]])
                states = np.expand_dims(states,-1)
                rewards = np.array([np.array(x) for x in data[:,1]])
                rewards = np.expand_dims(rewards,-1)
                _, preturn_loss, lambda_preturn_loss = model.train_on_batch(states, rewards)
                
                max_lambda_preturn_loss = max(max_lambda_preturn_loss, lambda_preturn_loss)
                max_preturn_loss = max(max_preturn_loss, preturn_loss)
                preturn_loss_arr.append(preturn_loss)
                lambda_preturn_loss_arr.append(lambda_preturn_loss)
                
        if step_counter % 1000 == 0 and step_counter > 1:
            print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
            
            if step_counter > config.episode_length+config.batch_size:
                print("running mean % of max preturn loss: ", "%.2f" % (100*np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):])/max_preturn_loss), "\t\t", np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):]))
                print("running mean % of max lambda preturn loss: ", "%.2f" % (100*np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):])/max_lambda_preturn_loss), "\t\t", np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):]))
                predictron_result = model.predict([state])
                DQN_arr.append(dqn_agent.calculate_value_of_action(state, allowed_actions))
                predictron_lambda_arr.append(predictron_result[1])
                reward_episode_arr.append(reward_episode)
                
                print(predictron_result[0],predictron_result[1], reward_episode, DQN_arr[-1])
        
        # record the information for use again in the next training example
        mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    
    if my_sim.env.now > config.burnin:
        step_counter += 1
            
    if TRAIN_DQN and step_counter >= config.DQN_train_steps:
        TRAIN_DQN = False
        step_counter = 0
    elif not TRAIN_DQN and step_counter >= config.Predictron_train_steps:
        TRAIN_DQN = True
        step_counter = 0

dqn_agent.save_model("DQN_predictron_full_"+str(args.state_rep_size)+".h5")
predictron.model.save(args.predictron_model_dir+'Predictron_dqn_full_'+str(args.state_rep_size)+'.h5')

predictron_error = np.abs(np.array(predictron_lambda_arr)[:,0]-np.array(reward_episode_arr))
predictron_error_avg = [predictron_error[0]]
alpha = 0.1
for i in range(len(predictron_error)-1):
    predictron_error_avg.append(predictron_error_avg[i]*(1-alpha) + predictron_error[i+1]*alpha)

DQN_error = np.abs(np.array(DQN_arr)-np.array(reward_episode_arr))
DQN_error_avg = [DQN_error[0]]
for i in range(len(predictron_error)-1):
    DQN_error_avg.append(DQN_error_avg[i]*(1-alpha) + DQN_error[i+1]*alpha)

predictron_dqn_error_avg=[DQN_error[0] - predictron_error[0]]
for i in range(len(predictron_error)-1):
    predictron_dqn_error_avg.append(predictron_dqn_error_avg[i]*(1-alpha) + (DQN_error[i+1]-predictron_error[i+1])*alpha)

predictron_ratio_error = np.asarray(predictron_lambda_arr)[:,0] / (np.asarray(reward_episode_arr)+1e-18)
predictron_ratio_error_avg = [predictron_ratio_error[0]]
for i in range(len(predictron_error)-1):
    predictron_ratio_error_avg.append(predictron_ratio_error_avg[i]*(1-alpha) + predictron_ratio_error[i+1]*alpha)


plt.figure()
plt.plot(preturn_loss_arr)

plt.figure()
plt.plot(lambda_preturn_loss_arr)

plt.figure()
plt.plot(dqn_loss_arr)

plt.figure()
plt.plot(reward_episode_arr, '.', label='Target')
plt.plot(predictron_lambda_arr, '.', label='Predictron')
plt.plot(DQN_arr, '.', label='DQN')
plt.title("Value estimate")
plt.xlabel("Thousands of steps")
plt.legend(loc='lower center')

plt.figure()
plt.plot(predictron_error, '.', label='Predictron')
plt.plot(predictron_error_avg, label='Running average Predictron')
plt.plot(DQN_error, '.', label='DQN')
plt.plot(DQN_error_avg, label='Running average DQN')
plt.title("Absolute value estimate error")
plt.legend()

plt.figure()
plt.plot(DQN_error[0:100] - predictron_error[0:100], '.', label='DQN - Predictron')
plt.plot(predictron_dqn_error_avg[0:100], label='Running average')
plt.title("DQN_error - predictron_error (first 100.000 steps)")
plt.xlabel("Thousands of steps")
plt.legend()
plt.axhline(linewidth=1, color='grey')

plt.figure()
plt.plot(DQN_error - predictron_error, '.', label='DQN - Predictron')
plt.plot(predictron_dqn_error_avg, label='Running average')
plt.title("DQN_error - predictron_error")
plt.xlabel("Thousands of steps")
plt.legend()
plt.axhline(linewidth=1, color='grey')

plt.figure()
plt.plot(predictron_error, label='Predictron')
plt.plot(predictron_error_avg, label='Running average')
plt.title("Predictron_error")
plt.xlabel("Thousands of steps")
plt.legend()
plt.axhline(linewidth=1, color='grey')

plt.figure()
plt.plot(predictron_ratio_error, '.', label='Predictron')
# plt.plot(predictron_ratio_error_avg, label='Running average')
plt.title("Predictron error ratio")
plt.xlabel("Thousands of steps")
plt.legend()
plt.axhline(linewidth=1, color='grey')
plt.ylim((-1,2.5))

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
print(np.mean(my_sim.lateness[-10000:]))

cols = [mean_util, mean_inter, std_inter, coeff_var]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival'])
df = df.transpose()
df.to_csv(args.save_dir+'util'+id+'_srs_'+str(args.state_rep_size)+'.csv')
# print(df)

# # Plot the time taken to complete each wafer
plt.figure()
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()

