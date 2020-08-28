import pandas as pd
import argparse
import json
import math

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--load_file_dir", default='~/Documents/workspace/WDsim/data/', help="Path to load factory setup csv files")
parser.add_argument("--save_file_dir", default='./b40_setup/', help="Path to save factory setup json files")
args = parser.parse_args()

recipes = pd.read_csv(args.load_file_dir + 'recipes.csv')
machines = pd.read_csv(args.load_file_dir + 'machines.csv')

num_seq_steps = 40
num_waf = 4

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

def get_proc_time(A, B, LS, include_load, load, include_unload, unload):
    proc_t = A * num_waf + B * math.ceil(num_waf/LS)
    if include_load == -1:
        proc_t += load
    if include_unload == -1:
        proc_t += unload
    return proc_t

recipes = recipes.dropna()
recipe_dict = dict()
for ht in list(recipes.HT.unique()):
    temp = recipes.loc[recipes['HT'] == ht]
    if len(temp) > 1:
        ls = []
        for index, row in temp.iterrows():
            ls.append([row[2], get_proc_time(row[3], row[4], row[5], row[6], row[7], row[8], row[9])])
        d  = {ht:ls}
        recipe_dict.update(d)
    else:
        ls = []
        ls.append([row[2], get_proc_time(row[3], row[4], row[5], row[6], row[7], row[8], row[9])])
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
machine_dict.update({'BLUEM-6': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-7': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-8': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-9': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-10': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-11': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-12': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-13': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-14': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-15': 'BLUEMOVEN'})
machine_dict.update({'BLUEM-16': 'BLUEMOVEN'})
machine_dict.update({'Z660-14': 'Z66013'})
machine_dict.update({'Z660-15': 'Z66013'})
machine_dict.update({'Z660-16': 'Z66013'})
machine_dict.update({'Z660-17': 'Z66013'})
machine_dict.update({'Z660-18': 'Z66013'})
machine_dict.update({'INS-3006n1': 'SPOTCHECK LIFTOFF'})
machine_dict.update({'INS-3006n1': 'SPOTCHECK LIFTOFF'})
machine_dict.update({'INS-3006n3': 'SPOTCHECK LIFTOFF'})
machine_dict.update({'SST-8': 'HOTSST'})
machine_dict.update({'SST-9': 'HOTSST'})
machine_dict.update({'SST-10': 'HOTSST'})
machine_dict.update({'INS-3012n1': 'LEICA PHOTO'})
machine_dict.update({'WRKBAKE-02n1': 'WRINKLE BAKE'})
machine_dict.update({'INS-3015n1': 'LEICA ETCH'})
machine_dict.update({'EMERALD-3n1': 'EMERALD'})
machine_dict.update({'EMERALD-3n2': 'EMERALD'})
machine_dict.update({'BSETGAPCP2n1': 'GAPETCH'})
machine_dict.update({'BSETGAPCP2n2': 'GAPETCH'})

# print(len(machine_dict))
# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
# recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}
recipes = recipe_dict

# average lead time for each head type
head_types = recipes.keys()
lead_dict = {}

part_mix = {}

for ht in head_types:
    d = {ht:1900}
    lead_dict.update(d)

    w = {ht:1}
    part_mix.update(w)

break_repair_WIP = {}

break_repair_WIP['break_mean'] = 1e5
break_repair_WIP['repair_mean'] = 120
break_repair_WIP['n_batch_wip'] = 30

if not os.path.exists(args.save_file_dir):
    os.makedirs(args.save_file_dir)

with open(args.save_file_dir+'break_repair_wip.json', 'w') as fp:
    json.dump(break_repair_WIP, fp)

with open(args.save_file_dir+'machines.json', 'w') as fp:
    json.dump(machine_dict, fp)

with open(args.save_file_dir+'recipes.json', 'w') as fp:
    json.dump(recipes, fp)

with open(args.save_file_dir+'due_date_lead.json', 'w') as fp:
    json.dump(lead_dict, fp)

with open(args.save_file_dir+'part_mix.json', 'w') as fp:
    json.dump(part_mix, fp)



