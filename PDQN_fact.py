import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Use this to run on CPU only
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
import json
import queue
import DeepQNet
import argparse
import datetime

from predictron import Predictron, Replay_buffer

id = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--dqn_model_dir", default='./data/DQN_model_2020-09-07-10-12-34_seed_0.h5', help="Path to the DQN model")
parser.add_argument("--state_rep_size", default='128', help="Size of the state representation")
parser.add_argument("--predictron_type", default='complete', help="Path to the DQN model")
parser.add_argument("--sim_time", default=2e7, type=int, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='b20_setup/', help="Path to factory setup files")
parser.add_argument("--save_dir", default='data/', help="Path save models and log files in")
parser.add_argument("--seed", default=0, help="random seed")
parser.add_argument("--sample_rate", default=None, help="sample rate for the predictron")
args = parser.parse_args()

sim_time = args.sim_time

res_dir = args.save_dir+args.factory_file_dir+'pdqn/'+'/'+str(id)+'/'
res_path = res_dir+'pdqn_sim_time'+str(args.sim_time)+'srs'+str(args.state_rep_size)+'seed'+str(args.seed)

# model_dir = args.save_dir+'models/srs_'+str(args.state_rep_size)+'/'+str(id)+'/'
pretrained_dqn_model_dir = args.dqn_model_dir
# predictron_model_dir = model_dir+'Predictron_'+args.predictron_type+'_srs_'+str(args.state_rep_size)+'.h5'
# dqn_model_dir = model_dir+'DQN_'+args.predictron_type+'_srs_'+str(args.state_rep_size)+'.h5'

# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

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

class Config_predictron():
    def __init__(self):
        # self.train_dir = './ckpts/predictron_train'
        # self.num_gpus = 1
        
        # adam optimizer:
        self.learning_rate = 1e-2
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-7
        
        self.l2_weight=0.01
        self.dropout_rate=0.
        
        # self.epochs = 5000
        self.batch_size = 32
        self.episode_length = 500
        self.predictron_update_rate = 500
        self.burnin = 1e4
        self.gamma = 0.99
        self.replay_memory_size = 100000
        self.predictron_update_steps = 50
        self.max_depth = 16
        
        self.DQN_train_steps = 5e4
        self.DQN_train_steps_initial = 5e4
        self.Predictron_train_steps = 5e4
        self.Predictron_train_steps_initial = 5e4
        self.train_itterations = 10
        
        if args.sample_rate:
            self.predictron_update_rate = args.sample_rate
            self.Predictron_train_steps = int(args.sample_rate * self.batch_size * 128 + self.episode_length)
            self.Predictron_train_steps_initial = int(args.sample_rate * self.batch_size * 128 + self.episode_length)
        
        
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
    return state_rep

config = Config_predictron()
if config.train_itterations is not None:
    sim_time=1e10
    print("simulation time set to 1e16 as itterations is set.")

# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, part_mix, break_repair_WIP['n_batch_wip'], break_mean=break_repair_WIP['break_mean'], repair_mean=break_repair_WIP['repair_mean'], seed=args.seed)
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
replay_buffer = Replay_buffer(memory_size = config.replay_memory_size, seed=args.seed)

predictron = Predictron(config)
model = predictron.model
preturn_loss_arr = []
max_preturn_loss = 0
lambda_preturn_loss_arr = []
max_lambda_preturn_loss = 0

DQN_arr =  []
predictron_lambda_arr = []
reward_episode_arr = []

# Creating the DQN agent
dqn_agent = DeepQNet.DQN(state_space_dim= state_size, action_space= action_space, epsilon_max=0.1, epsilon_min=0.1, gamma=0.99)
dqn_agent.load_model(pretrained_dqn_model_dir)
order_count = 0

TRAIN_DQN = False
step_counter = 0 # to count steps between itterations
itteration = 0
num_steps = 0 # to count all steps taken

dqn_loss_arr = []
pred_loss_arr = []
DQN_train_steps = config.DQN_train_steps_initial
Predictron_train_steps = config.Predictron_train_steps_initial

num_steps_total=0
if config.train_itterations is not None:
    num_steps_total = config.DQN_train_steps_initial+config.Predictron_train_steps_initial+config.train_itterations*(config.DQN_train_steps+config.Predictron_train_steps)

while (itteration is None and my_sim.env.now < sim_time) or (itteration is not None and itteration < config.train_itterations):
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
            if itteration is None:
                print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
            else:
                print(("%.2f" % (100*num_steps/num_steps_total))+"% done")
            print("Mean lateness: ", np.mean(my_sim.lateness))
        # if my_sim.order_completed:
        # After each wafer completed, train the policy network 
        loss = dqn_agent.replay(extern_target_model = predictron.model)
        if loss is not None:
            dqn_loss_arr.append(np.mean(loss))
        # order_count += 1
        # if order_count >= 1:
        #     # After every 20 processes update the target network and reset the order count
        #     dqn_agent.train_target()
        #     order_count = 0
        #     # Record the information for use again in the next training example
                
        mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    
    else:  #Predictron
        
        state_episode = state_queue.pop(0)
        state_queue.append(state)
                
        reward_queue = [config.gamma*x + reward for x in reward_queue]
        reward_episode = reward_queue.pop(0)
        reward_queue.append(0.)
        if step_counter > config.episode_length and step_counter % config.predictron_update_rate == 0:
            replay_buffer.put((state_episode, reward_episode))
            # if step_counter > config.episode_length+config.batch_size and my_sim.order_completed: # and (step_counter % config.predictron_update_steps) == 0:
                
                # data = np.array(replay_buffer.get(config.batch_size))
                # states = np.array([np.array(x) for x in data[:,0]])
                # states = np.expand_dims(states,-1)
                # rewards = np.array([np.array(x) for x in data[:,1]])
                # rewards = np.expand_dims(rewards,-1)
                # _, preturn_loss, lambda_preturn_loss = model.train_on_batch(states, rewards)
                
                # max_lambda_preturn_loss = max(max_lambda_preturn_loss, lambda_preturn_loss)
                # max_preturn_loss = max(max_preturn_loss, preturn_loss)
                # preturn_loss_arr.append(preturn_loss)
                # lambda_preturn_loss_arr.append(lambda_preturn_loss)
                
        if step_counter % 1000 == 0 and step_counter > 1:
            if itteration is None:
                print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
            else:
                print(("%.2f" % (100*num_steps/num_steps_total))+"% done")
            print("Mean lateness: ", np.mean(my_sim.lateness))
            
            # if step_counter > config.episode_length+config.batch_size:
            #     print("running mean % of max preturn loss: ", "%.2f" % (100*np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):])/max_preturn_loss), "\t\t", np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):]))
            #     print("running mean % of max lambda preturn loss: ", "%.2f" % (100*np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):])/max_lambda_preturn_loss), "\t\t", np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):]))
            #     predictron_result = model.predict([state])
            #     DQN_arr.append(dqn_agent.calculate_value_of_action(state, allowed_actions))
            #     predictron_lambda_arr.append(predictron_result[1])
            #     reward_episode_arr.append(reward_episode)
                
            #     print(predictron_result[0],predictron_result[1], reward_episode, DQN_arr[-1])            
            
        
        # record the information for use again in the next training example
        mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    
    num_steps += 1
    if my_sim.env.now > config.burnin:
        step_counter += 1
            
    if TRAIN_DQN and step_counter >= DQN_train_steps:
        DQN_train_steps = config.DQN_train_steps
        TRAIN_DQN = False
        step_counter = 0
        itteration += 1
        if args.sample_rate:
            dqn_agent.save_model(res_path+'pdqn_model_itt_'+str(itteration)+'_sr_'+str(config.predictron_update_rate)+'.h5')
            np.savetxt(res_path+'lateness_itt_'+str(itteration)+'_sr_'+str(config.predictron_update_rate)+'.csv', np.array(my_sim.lateness), delimiter=',')
        else:
            dqn_agent.save_model(res_path+'pdqn_model_itt_'+str(itteration)+'.h5')
            np.savetxt(res_path+'lateness_itt_'+str(itteration)+'.csv', np.array(my_sim.lateness), delimiter=',')
        print("Training predictron")
    elif not TRAIN_DQN and step_counter >= Predictron_train_steps:
        data = np.array(replay_buffer.get_pop(config.batch_size))
        while data != []:
            states = np.array([np.array(x) for x in data[:,0]])
            states = np.expand_dims(states,-1)
            rewards = np.array([np.array(x) for x in data[:,1]])
            rewards = np.expand_dims(rewards,-1)
            _, preturn_loss, lambda_preturn_loss = model.train_on_batch(states, rewards)
            
            max_lambda_preturn_loss = max(max_lambda_preturn_loss, lambda_preturn_loss)
            max_preturn_loss = max(max_preturn_loss, preturn_loss)
            preturn_loss_arr.append(preturn_loss)
            lambda_preturn_loss_arr.append(lambda_preturn_loss)
            
            print("running mean % of max preturn loss: ", "%.2f" % (100*np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):])/max_preturn_loss), "\t\t", np.mean(preturn_loss_arr[-min(10, len(preturn_loss_arr)):]))
            print("running mean % of max lambda preturn loss: ", "%.2f" % (100*np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):])/max_lambda_preturn_loss), "\t\t", np.mean(lambda_preturn_loss_arr[-min(10, len(lambda_preturn_loss_arr)):]))
            predictron_result = model.predict([state])
            DQN_arr.append(dqn_agent.calculate_value_of_action(state, allowed_actions))
            predictron_lambda_arr.append(predictron_result[1])
            reward_episode_arr.append(reward_episode)
            
            print(predictron_result[0],predictron_result[1], reward_episode, DQN_arr[-1])
            data = np.array(replay_buffer.get_pop(config.batch_size))
            
        
        replay_buffer.clear()
        Predictron_train_steps = config.Predictron_train_steps
        TRAIN_DQN = True
        step_counter = 0
        print("Training PDQN")
    
    
    if (DQN_train_steps == 0 and Predictron_train_steps == 0):
        break
# Save the trained DQN policy network
dqn_agent.save_model(res_path+'pdqn_model.h5')
predictron.model.save(res_path+'p_model.h5')

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
#
print("Mean lateness of last 10000 minutes", np.mean(my_sim.lateness[-10000:]))

cols = [mean_util, mean_inter, std_inter, coeff_var]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival'])
df = df.transpose()
df.to_csv(res_path+'util.csv')
# print(df)

np.savetxt(res_path+'lateness.csv', np.array(my_sim.lateness), delimiter=',')

# figure_dir = model_dir+"figures/"
# if not os.path.exists(figure_dir):
#     os.makedirs(figure_dir)
# 
# plt.figure()
# plt.plot(preturn_loss_arr)
# plt.savefig(figure_dir+"preturn_loss.png",dpi=600)
#
# plt.figure()
# plt.plot(lambda_preturn_loss_arr)
# plt.savefig(figure_dir+"lambda_preturn_loss.png",dpi=600)
#
# N=5000
# dqn_loss_arr_avg = np.convolve(dqn_loss_arr, np.ones((N,))/N, mode='valid')
# plt.figure()
# plt.plot(dqn_loss_arr)
# plt.plot(dqn_loss_arr_avg)
# plt.ylim((0,5000))
# plt.savefig(figure_dir+"dqn_loss.png",dpi=600)
#
# plt.figure()
# plt.plot(reward_episode_arr, '.', label='Target')
# plt.plot(predictron_lambda_arr, '.', label='Predictron')
# plt.plot(DQN_arr, '.', label='PDQN')
# plt.title("Value estimate")
# plt.xlabel("Thousands of steps")
# plt.legend(loc='lower center')
# plt.savefig(figure_dir+"value_estimate.png",dpi=600)
#
# plt.figure()
# plt.plot(predictron_error, '.', label='Predictron')
# plt.plot(predictron_error_avg, label='Running average Predictron')
# plt.plot(DQN_error, '.', label='PDQN')
# plt.plot(DQN_error_avg, label='Running average PDQN')
# plt.title("Absolute value estimate error")
# plt.legend()
# plt.savefig(figure_dir+"absolute_error.png",dpi=600)
#
# plt.figure()
# plt.plot(DQN_error[0:100] - predictron_error[0:100], '.', label='DQN - Predictron')
# plt.plot(predictron_dqn_error_avg[0:100], label='Running average')
# plt.title("DQN_error - predictron_error (first 100.000 steps)")
# plt.xlabel("Thousands of steps")
# plt.legend()
# plt.axhline(linewidth=1, color='grey')
#
# plt.figure()
# plt.plot(DQN_error - predictron_error, '.', label='DQN - Predictron')
# plt.plot(predictron_dqn_error_avg, label='Running average')
# plt.title("DQN_error - predictron_error")
# plt.xlabel("Thousands of steps")
# plt.legend()
# plt.axhline(linewidth=1, color='grey')
# plt.savefig(figure_dir+"DQN_predictron_error_dif.png",dpi=600)
#
# plt.figure()
# plt.plot(predictron_error, label='Predictron')
# plt.plot(predictron_error_avg, label='Running average')
# plt.title("Predictron_error")
# plt.xlabel("Thousands of steps")
# plt.legend()
# plt.axhline(linewidth=1, color='grey')
# plt.savefig(figure_dir+"predictron_error.png",dpi=600)
#
# plt.figure()
# plt.plot(predictron_ratio_error, '.', label='Predictron')
# # plt.plot(predictron_ratio_error_avg, label='Running average')
# plt.title("Predictron error ratio")
# plt.xlabel("Thousands of steps")
# plt.legend()
# plt.axhline(linewidth=1, color='grey')
# plt.ylim((-1,2.5))
# # plt.savefig(model_dir+"results/preturn_loss.png",dpi=600)
#
# # Total wafers produced
# # print("Total wafers produced:", len(my_sim.cycle_time))
# # # # i = 0
# # for ht in my_sim.recipes.keys():
# #     # for sequ in range(len(my_sim.recipes[ht])-1):
# #     # i += 1
# #     # print(len(my_sim.recipes[ht]))
# #     # waf = fact_sim.wafer_box(my_sim, 4, ht, my_sim.wafer_index, lead_dict, sequ)
# #     # my_sim.wafer_index += 1
# #     sequ = len(my_sim.recipes[ht])-1
# #     print(ht)
# #     print(sequ)
# #     print(my_sim.get_rem_shop_time(ht, sequ, 4))
#
# # print(my_sim.get_proc_time('ASGA', 99, 4))
# # print(i)
#
#
# # # Plot the time taken to complete each wafer
# plt.figure()
# plt.plot(my_sim.lateness, '.')
# plt.xlabel("Wafers")
# plt.ylabel("Lateness")
# plt.title("The amount of time each wafer was late")
# plt.xlim((25000,30000))
# plt.savefig(figure_dir+"wafer_lateness.png",dpi=600)

