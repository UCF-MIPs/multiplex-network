import pandas as pd
import numpy as np
from src import cos_sim
import os

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



# Run cosine similarity on heatmap dfs across datasets
skrip_noninfl_dist = cos_sim.cos_sim(skrip_sources, pure_noninfl_df)
skrip_disinfo_dist = cos_sim.cos_sim(skrip_sources, pure_disinfo_df)
#skrip_disinfo_dist = cos_sim.cos_sim(skrip_sources, mixed_disinfo_df)

ukr_noninfl_dist = cos_sim.cos_sim(ukr_sources, pure_noninfl_df)
ukr_disinfo_dist = cos_sim.cos_sim(ukr_sources, pure_disinfo_df)
#ukr_disinfo_dist = cos_sim.cos_sim(ukr_sources, mixed_disinfo_df)


print('skrip noninfl dist')
print(skrip_noninfl_dist)
print('ukr noninfl dist')
print(ukr_noninfl_dist)

print('skrip disinfo dist')
print(skrip_disinfo_dist)
print('ukr disinfo dist')
print(ukr_disinfo_dist)



