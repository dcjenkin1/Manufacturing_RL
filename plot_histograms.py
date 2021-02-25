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



# DQN_dir_list = ("data/b20_setup/pdqn/2020-10-02-14-37-09/pdqn/",)
# DQN_dir_list = ("data/b20_setup/pdqn/2020-10-02-14-37-44/pdqn/",)
DQN_dir_list = ("data/b20_setup/pdqn/2020-10-03-15-00-58/pdqn/",
                "data/b20_setup/pdqn/2020-10-02-14-40-28/pdqn/",
                "data/b20_setup/pdqn/2020-10-02-14-41-02/pdqn/",)

DQN_dir_list = ("data/b20_setup/pdqn/2020-09-21-11-18-07/pdqn/",
                "data/b20_setup/pdqn/2020-09-21-11-18-12/pdqn/",
                "data/b20_setup/pdqn/2020-09-21-11-18-17/pdqn/",
                "data/b20_setup/pdqn/2020-09-21-11-18-22/pdqn/",)

DQN_dir_list = ("data/b20_setup/pdqn/2020-10-06-22-54-45/pdqn/",
                "data/b20_setup/pdqn/2020-10-06-22-55-10/pdqn/",
                "data/b20_setup/pdqn/2020-10-06-22-55-30/pdqn/",
                "data/b20_setup/pdqn/2020-10-08-21-29-17/pdqn/",
                "data/b20_setup/pdqn/2020-10-08-21-30-09/pdqn/",
                "data/b20_setup/pdqn/2020-10-06-22-57-16/pdqn/",
                "data/b20_setup/pdqn/2020-10-06-22-59-06/pdqn/",)


# DQN_dir_list = ("pdqn/models/testdir/mymodel.h5",)
    

EXT = "wafer_lateness.csv"

mydata = []
ur=None
for model_dir in DQN_dir_list:
    mydata = []
    PATH,model_name=os.path.split(model_dir)

    all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
    

    for csv_file_path in all_csv_files:
        seed = re.findall(r'^\D*(\d+)', csv_file_path.partition('seed_')[-1])[0]
        itteration = re.findall(r'^\D*(\d+)', csv_file_path.partition('itt_')[-1])[0]
        sr = re.findall(r'^\D*(\d+)', csv_file_path.partition('ur_')[-1])[0]
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            lateness = [float(row[0]) for row in csv_reader]
            mean_lateness = np.mean(lateness[-10000:])
            max_lateness = np.max(lateness[-10000:])
            var_lateness = np.std(lateness[-10000:])
            # print(itteration, seed, mean_lateness, var_lateness, len(lateness), max(lateness))
            mydata.append([int(seed), int(itteration), mean_lateness, var_lateness, max_lateness, len(lateness)])
            
            plt.figure()
            data = lateness[-10000:]
            # X = np.array(data)
            # X = X[X!=0.0]
            binwidth = 10
            plt.hist(data,range(int(min(data)), int(max(data) + binwidth), binwidth))#, histtype=u'step', density=True)
            plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=1)
            plt.yscale('log')
            plt.xlim(-10,1500)
            min_ylim, max_ylim = plt.ylim()
            plt.text(np.mean(data)*1.1, max_ylim*0.5, 'Mean: {:.2f}'.format(np.mean(data)))
            plt.title("Sample rate: "+str(sr)+" Itteration "+str(itteration))
            plt.show()
            plt.pause(0.05) 

    mydata = np.array(mydata)
    if sr:
        print("Sample rate: "+str(sr))
        
    for myitt in np.unique(mydata[:,1]):
        # print("Iteration: "+str(myitt),mydata[:,2][mydata[:,1]==myitt])
        my_mean_lateness = np.mean(mydata[:,2][mydata[:,1]==myitt])
        my_mean_max_lateness = np.mean(mydata[:,4][mydata[:,1]==myitt])
        my_mean_std = np.mean(mydata[:,3][mydata[:,1]==myitt])
        my_mean_std_mean_lateness = np.std(mydata[:,2][mydata[:,1]==myitt])
        my_mean_throughput = np.mean(mydata[:,5][mydata[:,1]==myitt])
        print("Iter: "+ str(int(myitt)), "\tMean lateness: "+ "%.2f" % my_mean_lateness, "\tMean std: "+ "%.2f" % my_mean_std, "\tMean std of mean lateness: "+"%.2f" %  my_mean_std_mean_lateness,  "\tMean max lateness: "+"%.2f" %  my_mean_max_lateness, "\t Mean Throughput: "+"%.2f" %  my_mean_throughput)
    print('')