import factory_sim_throughput as fact_sim
import numpy as np
import pandas as pd
import math 
# import matplotlib
import random
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain
import argparse
import datetime
import json
from scipy.ndimage.filters import uniform_filter1d

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--sim_time", default=2e6, type=int, help="Simulation minutes")
parser.add_argument("--factory_file_dir", default='./b20_setup/', help="Path to factory setup files")
parser.add_argument("--save_dir", default='./pdqn/', help="Path save log files in")
parser.add_argument("--seed", default=0, help="random seed")
parser.add_argument("--method", default='cr', help="methods available: due_time, cr, fifo, randact_sorted, randact_unsorted, randact_rand, randwaf")
args = parser.parse_args()

id = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())# str(int(np.ceil(random.random()*10000)))
# random.seed(args.seed)
args = parser.parse_args()
s = args.save_dir

sim_time = args.sim_time

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
        wafer = wafer_list[0]
        return wafer
    else:
        if args.method=='cr':
            cr_ratio = {}
            for waf in wafer_list:
                
                cr_ = (waf.due_time - sim.env.now) / (sim.get_rem_shop_time(waf.HT, waf.seq))
                cr_ratio[waf] = cr_
            wafer = min(cr_ratio, key=cr_ratio.get)
            # cr = (wafer.due_time - sim.env.now) / (sim.get_rem_shop_time(wafer.HT, wafer.seq))
            # if  cr< 0:
            #     print("Late wafer selected", sim.env.now, wafer.due_time, cr, wafer)
            #     print(cr_ratio)
            #     print()
            
        elif args.method=='due_time':
            due_time = {}
            for waf in wafer_list:
                dt_ = waf.due_time
                due_time[waf] = dt_
            wafer = min(due_time, key=due_time.get)
        
        elif args.method=='fifo':
            wafer = wafer_list[0]
            
        elif args.method=='randact_sorted':
            allowed_actions = list(set([(waf.HT,waf.seq) for waf in wafer_list]))
            action = random.choice(allowed_actions)
            # print(wafer_list)
            wafer = next(waf for waf in sorted(wafer_list, key=lambda x: x.due_time) if waf.HT == action[0] and waf.seq == action[1])
            
        elif args.method=='randact_unsorted':
            # action = random.choice(my_sim.allowed_actions)
            allowed_actions = list(set([(waf.HT,waf.seq) for waf in wafer_list]))
            
            action = random.choice(allowed_actions)
            wafer = next(waf for waf in wafer_list if waf.HT == action[0] and waf.seq == action[1])
        
        elif args.method=='randact_rand':
            # action = random.choice(my_sim.allowed_actions)
            allowed_actions = list(set([(waf.HT,waf.seq) for waf in wafer_list]))
            
            action = random.choice(allowed_actions)
            wafer = random.choice([waf for waf in wafer_list if waf.HT == action[0] and waf.seq == action[1]])
        
        elif args.method=='randwaf':
            wafer = random.choice(wafer_list)
        else:
            print("Unknown method: "+args.method)
            print("Using fifo instead")
            wafer = wafer_list[0]
        
        return wafer
    

# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, lead_dict, part_mix, break_repair_WIP['n_batch_wip'],
                             break_mean=break_repair_WIP['break_mean'], repair_mean=break_repair_WIP['repair_mean'], seed=args.seed)
# start the simulation
my_sim.start()
# Retrieve machine object for first action choice
mach = my_sim.next_machine
# The action space is a list of tuples of the form [('ht1',0), ('ht1',1), ..., ('ht2', 0), ...] indicating the head
# types and sequence steps for all allowed actions.
action_space = list(chain.from_iterable(my_sim.station_HT_seq.values()))
action_size = len(action_space)

step_counter=0
all_reward = []
while my_sim.env.now < sim_time:
    # print(my_sim.env.now)
    
    wafer = choose_action(my_sim)

    my_sim.run_action(mach, wafer)
    # print('Step Reward:'+ str(my_sim.step_reward))
    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    # next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward
    all_reward.append(reward)
    # Update the machines and allowed actions for the next step
    mach, allowed_actions = next_mach, next_allowed_actions
    # print("State:", state)
    
    if step_counter % 10000 == 0 and step_counter > 1:
        print(("%.2f" % (100*my_sim.env.now/sim_time))+"% done")
        print("Mean lateness: ", np.mean(my_sim.lateness))
        print("Running mean lateness: ", np.mean(my_sim.lateness[-min(len(my_sim.lateness),10000):]))
        print("Running mean reward: ", np.mean(all_reward[-min(len(all_reward),10000):]))
    step_counter+=1

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
machines_per_station = {station: len([mach for mach in my_sim.machines_list if mach.station == station]) for station in
                        my_sim.stations}

# print(np.mean(my_sim.lateness[-1000:]))

cols = [mean_util, mean_inter, std_inter, coeff_var, machines_per_station, station_wait_times]
df = pd.DataFrame(cols, index=['mean_utilization', 'mean_interarrival_time', 'standard_dev_interarrival',
                  'coefficient_of_var_interarrival', 'machines_per_station', 'mean_wait_time'])
df = df.transpose()
df.to_csv(args.save_dir+'util'+id+'seed'+str(args.seed)+'.csv')

np.savetxt(args.save_dir+'wafer_lateness'+id+'seed'+str(args.seed)+'.csv', np.array(my_sim.lateness), delimiter=',')

# # Plot the time taken to complete each wafer
plt.plot(my_sim.lateness)
plt.xlabel("Wafers")
plt.ylabel("Lateness")
plt.title("The amount of time each wafer was late")
plt.show()

N = 10000
x = my_sim.lateness
y = uniform_filter1d(x, size=N)
plt.plot(y)
plt.xlabel("Wafers")
plt.ylabel("Average lateness")
plt.title("The running average of the time each wafer was late, avg of "+str(N)+" wafers")
plt.show()

#
# Plot the time taken to complete each wafer
plt.plot(my_sim.cumulative_reward_list)
plt.xlabel("step")
plt.ylabel("Cumulative Reward")
plt.title("The sum of all rewards up until each time step")
plt.show()

data = my_sim.lateness
binwidth = 2
plt.hist(data,range(int(min(data)), int(max(data) + binwidth), binwidth))#, histtype=u'step', density=True)
plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=1)
plt.yscale('log')
# plt.xlim(-10,1500)
# min_ylim, max_ylim = plt.ylim()
plt.show()

data = my_sim.lateness[-10000:]
binwidth = 2
plt.hist(data,range(int(min(data)), int(max(data) + binwidth), binwidth))#, histtype=u'step', density=True)
plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=1)
plt.yscale('log')
# plt.xlim(-10,1500)
# min_ylim, max_ylim = plt.ylim()
plt.show()
