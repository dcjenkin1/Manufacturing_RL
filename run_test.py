# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:03:15 2020

@author: RTS
"""
import os, time, random
from threading import Thread
from queue import Queue

num_seeds=10

DQN_dir_list = ("DQN_model_5e5.h5",)
#                 "data/models/srs_1/2020-08-16-14-58-18/DQN_complete_srs_1.h5",\
#                 "data/models/srs_2/2020-08-14-15-05-06/DQN_complete_srs_2.h5",\
#                 "data/models/srs_3/2020-08-14-22-43-16/DQN_complete_srs_3.h5",\
#                 "data/models/srs_4/2020-08-14-15-05-21/DQN_complete_srs_4.h5",\
#                 "data/models/srs_8/2020-08-14-15-05-26/DQN_complete_srs_8.h5",\
#                 "data/models/srs_12/2020-08-17-11-01-51/DQN_complete_srs_12.h5",\
#                 "data/models/srs_16/2020-08-14-15-05-35/DQN_complete_srs_16.h5",\
#                 "data/models/srs_32/2020-08-14-15-05-41/DQN_complete_srs_32.h5",\
#                 "data/models/srs_128/2020-08-14-15-06-05/DQN_complete_srs_128.h5",\
#                 "data/models/srs_256/2020-08-14-15-06-14/DQN_complete_srs_256.h5",\
#                 "data/models/srs_16/2020-08-21-08-04-47/DQN_complete_srs_16.h5",\
#                 "data/models/srs_32/2020-08-21-08-04-59/DQN_complete_srs_32.h5",\
#                 "data/models/srs_128/2020-08-21-08-05-17/DQN_complete_srs_128.h5",\
#                 "data/models/srs_256/2020-08-21-08-05-32/DQN_complete_srs_256.h5",\
#                 "data/models/srs_1/2020-08-17-08-54-52/DQN_complete_srs_1.h5",\
#                 "data/models/srs_2/2020-08-17-08-55-32/DQN_complete_srs_2.h5",\
#                 "data/models/srs_3/2020-08-17-08-55-36/DQN_complete_srs_3.h5",\
#                 "data/models/srs_4/2020-08-17-08-55-46/DQN_complete_srs_4.h5",\
#                 "data/models/srs_8/2020-08-17-08-55-53/DQN_complete_srs_8.h5",\
#                 "data/models/srs_12/2020-08-17-08-57-54/DQN_complete_srs_12.h5",\
#                 "data/models/srs_16/2020-08-17-08-55-56/DQN_complete_srs_16.h5",\
#                 "data/models/srs_32/2020-08-17-08-56-00/DQN_complete_srs_32.h5",\
#                 "data/models/srs_128/2020-08-17-08-56-06/DQN_complete_srs_128.h5",\
#                 "data/models/srs_256/2020-08-17-08-56-21/DQN_complete_srs_256.h5",\
#                 "data/models/srs_16/2020-08-21-10-33-04/DQN_complete_srs_16.h5",\
#                 "data/models/srs_32/2020-08-21-10-33-11/DQN_complete_srs_32.h5",\
#                 "data/models/srs_128/2020-08-21-10-33-20/DQN_complete_srs_128.h5",\
#                 "data/models/srs_256/2020-08-21-10-33-37/DQN_complete_srs_256.h5",) 
                

# PDN_dir_list = ("PDQN_500000_full_16.h5",\
#                 "PDQN_500000_full_32.h5",\
#                 "PDQN_500000_full_128.h5",\
#                 "PDQN_500000_full_256.h5",)

def worker():
    while True:
        item = q.get()
        try:
            os.system(item)
        except:
            print("An exception occurred in "+ item)
        q.task_done()

q = Queue()
for i in DQN_dir_list:#+PDN_dir_list:
    t = Thread(target=worker)
    t.setDaemon(True)
    t.start()

for seed in range(num_seeds):
    for item in DQN_dir_list:
        item = "python rollout_DQN.py --model_dir="+item+" --seed="+str(seed)
        print(item)
        q.put(item)
    for item_ in PDN_dir_list:
        item = "python rollout_PDQN.py --model_dir="+item_+" --seed="+str(seed)
        print(item)
        q.put(item)
        
q.join()
