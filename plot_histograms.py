# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:37:39 2020

@author: RTS
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:51:12 2020

@author: RTS
"""
import os, re
import csv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt



DQN_dir_list = ("data/b20_setup/pdqn/2020-09-21-11-18-12/pdqn/seed_0",)
                

# DQN_dir_list = ("pdqn/models/testdir/mymodel.h5",)
    

EXT = "*wafer_lateness*.csv"


for model_dir in DQN_dir_list:
    PATH,model_name=os.path.split(model_dir)

    all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
    
    itteration =0
    for csv_file_path in all_csv_files:
        seed = re.findall(r'^\D*(\d+)', csv_file_path.partition('seed_')[-1])[0]
        itteration += 1
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            lateness = [float(row[0]) for row in csv_reader]
            mean_lateness = np.mean(lateness)
            var_lateness = np.var(lateness)
            print(itteration, seed, mean_lateness, var_lateness, len(lateness), max(lateness))
            plt.figure()
            data = lateness#[-10000:]
        
            binwidth = 10
            plt.hist(data,range(int(min(data)), int(max(data) + binwidth), binwidth))#, histtype=u'step', density=True)
            plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=1)
            plt.yscale('log')
            plt.xlim(-10,1500)
            min_ylim, max_ylim = plt.ylim()
            plt.text(np.mean(data)*1.1, max_ylim*0.5, 'Mean: {:.2f}'.format(np.mean(data)))
            plt.title("Itteration "+str(itteration))