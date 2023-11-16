import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from src import cos_sim

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# Get heatmap dfs (load from csvs, np objects, pkls)
datasets = ['Skripal', 'Ukraine'] # options: (include scenarios)

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat', 'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland', 'tanks','Zelensky_visit_the_US']

heatmap_dfs = {}

for name in dataset_list:
    # check if pkl file exists
    filename = f'results/quadheatmaps/{name}_sources_heatmap_df.csv'
    #filename = f'results/RQ2_{name}_heatmap_df.pkl'
    if os.path.exists(filename):
        print(f'loading {name} quadrant correlation heatmap dataframe')
        #df_heatmap = pd.read_pickle(f'{filename}')
        #df_heatmap = pd.read_csv(f'{filename}.csv')
        df_heatmap = pd.read_csv(filename, index_col=0)
    else:
        print(f'not found - {filename}')
    heatmap_dfs[name] = df_heatmap
    
skrip_sources = heatmap_dfs['Skripal']
ukr_sources = heatmap_dfs['Ukraine']

print(skrip_sources)
print(ukr_sources)

# Check heatmap dfs for labels (existence and order)
#TODO

#test cos_sim value by comparing two datasets, then each one seperately against artificial baseline
cs = cos_sim.cos_sim(skrip_sources, ukr_sources)
print(cs)


# artificial baselines
# TODO replace these placeholders, "disinfo" could be defined as U->T
# Turned arrays into df's because cos_sim function only takes in dfs

# labels:
#     TM TF UM UF
# TM
# TF
# UM
# UF
pure_noninfl = np.array([1.0, 0.0, 0.0, 0.0, \
                           0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0])
pure_noninfl_df = pd.DataFrame(pure_noninfl)

pure_disinfo = np.array([1.0, 0.0, 0.0, 0.0, \
                           0.0, 1.0, 0.0, 0.0, \
                           1.0, 1.0, 1.0, 0.0, \
                           1.0, 1.0, 0.0, 1.0])
pure_disinfo_df = pd.DataFrame(pure_disinfo)

mixed_infl = np.array([1.0, 0.5, 0.5, 0.5, \
                           0.5, 1.0, 0.5, 0.5, \
                           0.5, 0.5, 1.0, 0.5, \
                           0.5, 0.5, 0.5, 1.0])
mixed_infl_df = pd.DataFrame(mixed_infl)

distance_df = pd.DataFrame()
distance_labels = ['similarity_to_noninfl', 'similarity_to_disinfo']
zero_data = np.zeros(shape=(len(dataset_list), len(distance_labels)))
dist_df = pd.DataFrame(zero_data, index=dataset_list, columns=distance_labels)

for dataset in dataset_list:
    print(f'{dataset}')
    heatdf = heatmap_dfs[dataset]
    
    cs1 = cos_sim.cos_sim(heatdf, pure_noninfl_df)
    print(f'distance from non-influence: {cs1}')
    dist_df.at[dataset,'similarity_to_noninfl'] = cs1
    
    cs2 = cos_sim.cos_sim(heatdf, pure_disinfo_df)
    dist_df.at[dataset,'similarity_to_disinfo'] = cs2
    print(f'distance from disinformation: {cs2}')
    
    print('\n')

fig = dist_df.plot.scatter('similarity_to_noninfl','similarity_to_disinfo').get_figure()
ax2=fig.gca()
for k,v in dist_df.iterrows():
    ax2.annotate(k,v)

ax2.set_xlabel('Similarity to non-influence matrix')
ax2.set_ylabel('Similarity to disinformation matrix')

fig.savefig('results/datasets_cos_sim.png')

print(dist_df)
