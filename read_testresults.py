# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:51:12 2020

@author: RTS
"""
import os, re
import csv
from glob import glob
import numpy as np



DQN_dir_list = ("data/models/dqn/DQN_model_5e5.h5",\
                "data/models/srs_1/2020-08-16-14-58-18/DQN_complete_srs_1.h5",\
                "data/models/srs_2/2020-08-14-15-05-06/DQN_complete_srs_2.h5",\
                "data/models/srs_3/2020-08-14-22-43-16/DQN_complete_srs_3.h5",\
                "data/models/srs_4/2020-08-14-15-05-21/DQN_complete_srs_4.h5",\
                "data/models/srs_8/2020-08-14-15-05-26/DQN_complete_srs_8.h5",\
                "data/models/srs_12/2020-08-17-11-01-51/DQN_complete_srs_12.h5",\
                "data/models/srs_16/2020-08-14-15-05-35/DQN_complete_srs_16.h5",\
                "data/models/srs_32/2020-08-14-15-05-41/DQN_complete_srs_32.h5",\
                "data/models/srs_128/2020-08-14-15-06-05/DQN_complete_srs_128.h5",\
                "data/models/srs_256/2020-08-14-15-06-14/DQN_complete_srs_256.h5",\
                "data/models/srs_16/2020-08-21-08-04-47/DQN_complete_srs_16.h5",\
                "data/models/srs_32/2020-08-21-08-04-59/DQN_complete_srs_32.h5",\
                "data/models/srs_128/2020-08-21-08-05-17/DQN_complete_srs_128.h5",\
                "data/models/srs_256/2020-08-21-08-05-32/DQN_complete_srs_256.h5",\
                "data/models/srs_1/2020-08-17-08-54-52/DQN_complete_srs_1.h5",\
                "data/models/srs_2/2020-08-17-08-55-32/DQN_complete_srs_2.h5",\
                "data/models/srs_3/2020-08-17-08-55-36/DQN_complete_srs_3.h5",\
                "data/models/srs_4/2020-08-17-08-55-46/DQN_complete_srs_4.h5",\
                "data/models/srs_8/2020-08-17-08-55-53/DQN_complete_srs_8.h5",\
                "data/models/srs_12/2020-08-17-08-57-54/DQN_complete_srs_12.h5",\
                "data/models/srs_16/2020-08-17-08-55-56/DQN_complete_srs_16.h5",\
                "data/models/srs_32/2020-08-17-08-56-00/DQN_complete_srs_32.h5",\
                "data/models/srs_128/2020-08-17-08-56-06/DQN_complete_srs_128.h5",\
                "data/models/srs_256/2020-08-17-08-56-21/DQN_complete_srs_256.h5",\
                "data/models/srs_16/2020-08-21-10-33-04/DQN_complete_srs_16.h5",\
                "data/models/srs_32/2020-08-21-10-33-11/DQN_complete_srs_32.h5",\
                "data/models/srs_128/2020-08-21-10-33-20/DQN_complete_srs_128.h5",\
                "data/models/srs_256/2020-08-21-10-33-37/DQN_complete_srs_256.h5",) 
                

PDN_dir_list = ("data/PDN/models/srs_16/PDQN_500000_full_16.h5",\
                "data/PDN/models/srs_32/PDQN_500000_full_32.h5",\
                "data/PDN/models/srs_128/PDQN_500000_full_128.h5",\
                "data/PDN/models/srs_256/PDQN_500000_full_256.h5",)

# DQN_dir_list = ("data/models/testdir/mymodel.h5",)
    

EXT = "*wafer_lateness.csv"


for model_dir in DQN_dir_list+PDN_dir_list:
    PATH,model_name=os.path.split(model_dir)

    all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
    
    
    for csv_file_path in all_csv_files:
        seed = re.findall(r'^\D*(\d+)', csv_file_path.partition('seed_')[-1])[0]
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            lateness = [float(row[0]) for row in csv_reader]
            mean_lateness = np.mean(lateness)
            var_lateness = np.var(lateness)
            print(model_name, seed, mean_lateness, var_lateness)