import argparse
import json
import numpy as np
from collections import defaultdict
import scipy.stats as stats
import os
import datetime

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--factory_file_dir", default='b20_setup/', help="Path to factory setup files")
parser.add_argument("--save_file_dir", default='./r20_setup/', help="Path to save factory setup json files")
parser.add_argument("--seed", default=0, help="Path to save factory setup json files")
args = parser.parse_args()

np.random.seed(args.seed)

id = '{date:%Y-%m-%d-%H}'.format(date=datetime.datetime.now())

save_dir = args.save_file_dir+id

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

print('balanced')
print(break_repair_WIP)
print(machine_dict)
print(recipes)
print(lead_dict)
print(part_mix)

n_head_types = 10

n_seq_steps = 20

# mean_time_to_fail = 3360
#
# mean_time_to_repair = 30

num_stations = 24

fit_alpha, fit_loc, fit_beta = 0.8319306209481123, 1.2985144823146346, 26.17065110149774

dfit_alpha, dfit_loc, dfit_beta = 0.524386479195013, -1.0610137500367203e-28, 7.201557688154551

xfactor = 2

# mean_proc_t = 29
#
# std_dev_proc_time = 32

head_types = []
for i in range(n_head_types):
    head_types.append('ht' + str(i))

# ht_dem = {ht: stats.gamma.rvs(dfit_alpha, loc=dfit_loc, scale=dfit_beta, size=1)/4 for ht in head_types}

ht_dem = {ht: 10 for ht in head_types}

print(ht_dem)

stationsl = []
for i in range(num_stations):
    stationsl.append('st'+str(i))

recipesd = defaultdict(list)

for ht in head_types:
    for i in range(n_seq_steps):
        recipesd[ht].append([np.random.choice(stationsl), stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=1)[0]])

rrecipes = dict(recipesd)

tproct = {ht: 0 for ht in head_types}


for ht in head_types:
    for step in rrecipes[ht]:
        tproct[ht] += step[1]

htstatct = {ht: {stat: 0 for stat in stationsl} for ht in recipesd.keys()}
proc_ts = []
for ht in recipesd.keys():
    for step in recipesd[ht]:
        htstatct[ht][step[0]] += step[1]
        proc_ts.append(step[1])

htstatdem = {ht: {stat: 0 for stat in stationsl} for ht in head_types}

for ht in head_types:
    for stat in stationsl:
        htstatdem[ht][stat] = htstatct[ht][stat]*ht_dem[ht]

statdem = {stat: 0 for stat in stationsl}

for ht in head_types:
    for stat in stationsl:
        statdem[stat] += htstatdem[ht][stat]

statmachmpd = {stat: 1440 for stat in stationsl}

statmachdem = {}
for stat in stationsl:
    statmachdem[stat] = int(np.ceil(statdem[stat]/statmachmpd[stat]))

rmachine_dict = {}

for stat in stationsl:
    for i in range(statmachdem[stat]):
        rmachine_dict[stat + '-' + str(i)] = stat

rbreak_repair_WIP = {'break_mean': None, 'repair_mean': 102, 'n_batch_wip': 5}
rlead_dict = {}
for ht in head_types:
    rlead_dict[ht] = xfactor*tproct[ht]

rpart_mix = {}

for ht in head_types:
    rpart_mix[ht] = 1

print('random')

print(rbreak_repair_WIP)
print(rmachine_dict)
print(rrecipes)
print(rlead_dict)
print(rpart_mix)

with open(save_dir+'/break_repair_wip.json', 'w') as fp:
    json.dump(rbreak_repair_WIP, fp)

with open(save_dir+'/machines.json', 'w') as fp:
    json.dump(rmachine_dict, fp)

with open(save_dir+'/recipes.json', 'w') as fp:
    json.dump(rrecipes, fp)

with open(save_dir+'/due_date_lead.json', 'w') as fp:
    json.dump(rlead_dict, fp)

with open(save_dir+'/part_mix.json', 'w') as fp:
    json.dump(rpart_mix, fp)


print(len(rmachine_dict))
# fit_alpha, fit_loc, fit_beta=stats.gamma.fit(proc_ts)
#
# print(fit_alpha, fit_loc, fit_beta)
#
# da
#
#
# print(recipesd)
# # print(htstatct)
# #
# # print(break_repair_WIP)
# #
# #

# #

# #
# #
# #

# #
# # print(statmachdem)
# #
# # # machines_per_station = {station: len([mach for mach in my_sim.machines_list if mach.station == station]) for station in
# # #                         my_sim.stations}
# #
# # machines_at_station = {stat: [] for stat in stations}
# # for mach in machine_dict.keys():
# #     machines_at_station[machine_dict[mach]].append(mach)
# #
# # print(machines_at_station)
#
# machines_per_stat = {stat: len(machines_at_station[stat]) for stat in stations}
#
# print(machines_per_stat))
