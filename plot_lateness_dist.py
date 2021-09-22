import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os
import random

sns.set_theme(style="whitegrid")

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.bottom'] = False
# plt.rcParams['axes.spines.left'] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'stix'

MINI_SIZE = 8
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
figuresize = (8,2.5) #inches
lw = 1
ms = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MINI_SIZE)    # legend fontsize

plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure',figsize=figuresize)
plt.rc('figure',dpi=600)

def get_data(data_path='./data/lateres/',folder='b20'):
    path=data_path+folder
    all_files = [file
                 for path_, subdir, files in os.walk(path)
                 for file in glob.glob(os.path.join(path_, '*.csv'))]
    
    li = []
    # random.shuffle(all_files)
    for i,filename in enumerate(all_files):
        seed = re.findall(r"\d+", filename)[-1]
        if int(seed) >=2000:
            df = pd.read_csv(filename, index_col=None, header=None, names=['Lateness'])
            if 'cr' in filename:
                df['Method']='CR'
            elif 'fifo' in filename:
                df['Method']='FIFO'
            elif 'pdqn' in filename:
                df['Method']='PDQN'
            elif 'dqn' in filename:
                df['Method']='DQN'
            
            df['Seed'] = seed
            df['LastPart'] = False
            df[-10000:]['LastPart'] = True
            
            li.append(df)
        
        # if i > 20:
        #     break
    
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['Environment'] = folder
    return frame


data_path = './data/lateres/'
folders=os.listdir(data_path)

# folders = ['b20']
dfs = []
for folder in folders:
    dfs.append(get_data(data_path,folder))
#     df = dfs[-1]
#     title = folder
#     fig, axes = plt.subplots(nrows = 2, ncols=1, gridspec_kw={'height_ratios': [ 3, 1]},figsize=[8,8])
    
#     # sns.barplot(y=["cr", "dqn", "fifo", "pdqn"],
#     #             x=df.groupby(['Method'])['Lateness'].sum(),                 
#     #             order=["CR", "FIFO", "DQN", "PDQN"],
#     #             ax=axes[0],
#     #             edgecolor='k')
#     # axes[0].set(ylabel= "Method", xlabel = "$\sum$ Lateness")
#     # axes[0].set(ylabel= None, xlabel = None)
    
#     sns.violinplot(y="Method", 
#                     x="Lateness", 
#                     data=df, 
#                     order=["CR", "FIFO", "DQN", "PDQN"], 
#                     ax=axes[0],
#                     cut=0,
#                     inner=None,
#                     bw = 0.1)
    
#     sns.pointplot(x = df.groupby(['Method'])['Lateness'].mean(), 
#                   y=["CR", "DQN", "FIFO", "PDQN"],                
#                   order=["CR", "FIFO", "DQN", "PDQN"],
#                   ax=axes[0],
#                   join=False,
#                   marker='o',
#                   color='0.15',
#                   scale=0.6)
    
#     axes[0].set(ylabel= None, xlabel = None)
    
#     sns.boxplot(y="Method", 
#                 x="Lateness", 
#                 data=df, 
#                 order=["CR", "FIFO", "DQN", "PDQN"],
#                 showfliers = False,
#                 showmeans=True,
#                 meanprops={"marker":"o",
#                             "markerfacecolor":"0.15", 
#                             "markeredgecolor":"0.15",
#                           "markersize":"4"},
#                 ax=axes[1],
#                 whis=[2, 98])
#     axes[1].set(ylabel= None, xlabel = "Lateness")
    
#     # print(df.groupby(['Method'])['Lateness'].sum())
    
    
#     # fig.suptitle(title, fontsize=20)
#     plt.tight_layout()
#     plt.savefig('figures/'+folder+'.pdf', dpi=600)
#     plt.show()


df = pd.concat(dfs, ignore_index=True)


df['Environment'].loc[df.Environment=='wip15mf10000'] = 'G20\nWIP:15\nMTTF:10,000'
df['Environment'].loc[df.Environment=='wip15mf100000'] = 'G20\nWIP:15\nMTTF:100,000'
df['Environment'].loc[df.Environment=='wip30mf10000'] = 'G20\nWIP:30\nMTTF:10,000'
df['Environment'].loc[df.Environment=='wip30mf100000'] = 'G20\nWIP:30\nMTTF:100,000'

df_sum = df.groupby(['Seed','Environment','Method'],as_index=False)['Lateness'].sum()
df_mean = df.loc[df.LastPart].groupby(['Seed','Environment','Method'],as_index=False)['Lateness'].mean()

df_count = df.groupby(['Seed','Environment','Method'],as_index=False)['Lateness'].count()

fig, axes = plt.subplots(2,3,gridspec_kw={'width_ratios': [1, 2, 2]}, figsize=[8,5])

# sns.barplot(x="Environment", 
#             y='Lateness', 
#             hue='Method', 
#             hue_order=["CR", "FIFO", "DQN", "PDQN"], 
#             data=df_sum.loc[df_sum.Environment == 'B20'],
#             ax=axes[0,0],
#             edgecolor="0.15"
#             )
# sns.barplot(x="Environment", 
#             y='Lateness', 
#             hue='Method', 
#             hue_order=["CR", "FIFO", "DQN", "PDQN"], 
#             data=df_sum.loc[((df_sum.Environment=='G20\nWIP:15\nMTTF:10,000') | (df_sum.Environment=='G20\nWIP:15\nMTTF:100,000'))],
#             ax=axes[0,1],
#             edgecolor="0.15"
#             )
# sns.barplot(x="Environment", 
#             y='Lateness', 
#             hue='Method', 
#             hue_order=["CR", "FIFO", "DQN", "PDQN"], 
#             data=df_sum.loc[((df_sum.Environment=='G20\nWIP:30\nMTTF:10,000') | (df_sum.Environment=='G20\nWIP:30\nMTTF:100,000'))],
#             ax=axes[0,2],
#             edgecolor="0.15"
#             )


# axes[0,0].set(xlabel=None, ylabel = "$\sum$ Lateness")
# axes[0,1].set(xlabel=None, ylabel = None)
# axes[0,2].set(xlabel=None, ylabel = None)
# axes[0,0].get_legend().remove()
# axes[0,1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1),ncol=4)
# # axes[0,1].get_legend().remove()
# axes[0,2].get_legend().remove()

sns.boxplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_mean.loc[(df_mean.Environment == 'B20')],
            ax=axes[0,0],
            showfliers = False,
            whis=[2, 98],
            showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"0.85", 
                       "markeredgecolor":"0.15",
                       "markersize":"4"})
sns.boxplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_mean.loc[((df_mean.Environment=='G20\nWIP:15\nMTTF:10,000') | (df_mean.Environment=='G20\nWIP:15\nMTTF:100,000'))],
            ax=axes[0,1],
            showfliers = False,
            whis=[2, 98],
            showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"0.85", 
                       "markeredgecolor":"0.15",
                       "markersize":"4"})

sns.boxplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_mean.loc[((df_mean.Environment=='G20\nWIP:30\nMTTF:10,000') | (df_mean.Environment=='G20\nWIP:30\nMTTF:100,000'))],
            ax=axes[0,2],
            showfliers = False,
            whis=[2, 98],
            showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"0.85", 
                       "markeredgecolor":"0.15",
                       "markersize":"4"})

# axes[0,0].set(xticklabels=[])
# axes[0,1].set(xticklabels=[])
# axes[0,2].set(xticklabels=[])
axes[0,0].set(xlabel=None, ylabel = "Mean Lateness")
axes[0,1].set(xlabel=None, ylabel = None)
axes[0,2].set(xlabel=None, ylabel = None)
axes[0,0].get_legend().remove()
axes[0,1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1),ncol=4)
# axes[1,1].get_legend().remove()
axes[0,2].get_legend().remove()

sns.barplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_count.loc[df_count.Environment == 'B20'],
            ax=axes[1,0],
            edgecolor="0.15"
            )
sns.barplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_count.loc[((df_count.Environment=='G20\nWIP:15\nMTTF:10,000') | (df_count.Environment=='G20\nWIP:15\nMTTF:100,000'))],
            ax=axes[1,1],
            edgecolor="0.15"
            )
sns.barplot(x="Environment", 
            y='Lateness', 
            hue='Method', 
            hue_order=["CR", "FIFO", "DQN", "PDQN"], 
            data=df_count.loc[((df_count.Environment=='G20\nWIP:30\nMTTF:10,000') | (df_count.Environment=='G20\nWIP:30\nMTTF:100,000'))],
            ax=axes[1,2],
            edgecolor="0.15"
            )


axes[0,0].set(xticklabels=[])
axes[0,1].set(xticklabels=[])
axes[0,2].set(xticklabels=[])
axes[1,0].set(xlabel=None, ylabel = "Completed parts")
axes[1,1].set(xlabel=None, ylabel = None)
axes[1,2].set(xlabel=None, ylabel = None)
axes[1,0].get_legend().remove()
# axes[1,1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1),ncol=4)
axes[1,1].get_legend().remove()
axes[1,2].get_legend().remove()



plt.tight_layout()
plt.savefig('figures/'+'results'+'.pdf', dpi=600)
plt.show()
