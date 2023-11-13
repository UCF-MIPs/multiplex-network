import pandas as pd
import numpy as np
from src import cos_sim
import os

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# Get heatmap dfs (load from csvs, np objects, pkls)
names = ['Skripal', 'Ukraine'] # options: (include scenarios)

heatmap_dfs = {}

for name in names:
    # check if pkl file exists
    filename = f'results/RQ2_{name}_heatmap_df.pkl'
    if os.path.exists(filename):
        print(f'loading {name} heatmap dataframe')
        df_heatmap = pd.read_pickle(f'{filename}')
        #df_heatmap = pd.read_csv(f'{filename}.csv')
    else:
        print(f'not found - {filename}')
    heatmap_dfs[name] = df_heatmap
    
skrip_sources = heatmap_dfs['Skripal']
ukr_sources = heatmap_dfs['Ukraine']


# Check heatmap dfs for labels (existence and order)
#TODO

#test cos_sim value by comparing two datasets, then each one seperately against artificial baseline
cs = cos_sim.cos_sim(skrip_sources, ukr_sources)
print(cs)


# artificial baselines
# TODO replace these placeholders, "disinfo" could be defined as U->T
# Turned arrays into df's because cos_sim function only takes in dfs
pure_noninfl = np.array([1.0, 0.0, 0.0, 0.0, \
                           0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0])
pure_noninfl_df = pd.DataFrame(pure_noninfl)

pure_disinfo = np.array([1.0, 1.0, 1.0, 1.0, \
                           1.0, 1.0, 1.0, 1.0, \
                           1.0, 1.0, 1.0, 1.0, \
                           1.0, 1.0, 1.0, 1.0])
pure_disinfo_df = pd.DataFrame(pure_disinfo)

mixed_disinfo = np.array([1.0, 0.5, 0.5, 0.5, \
                           0.5, 1.0, 1.0, 0.5, \
                           0.5, 0.5, 0.5, 0.5, \
                           0.5, 0.5, 0.5, 1.0])
mixed_disinfo_df = pd.DataFrame(mixed_disinfo)



# Run cosine similarity on heatmap dfs across datasets
skrip_noninfo_dist = cos_sim.cos_sim(skrip_sources, pure_noninfl_df)
skrip_disinfo_dist = cos_sim.cos_sim(skrip_sources, pure_disinfo_df)
skrip_disinfo_dist = cos_sim.cos_sim(skrip_sources, mixed_disinfo_df)

ukr_noninfo_dist = cos_sim.cos_sim(ukr_sources, pure_noninfl_df)
ukr_disinfo_dist = cos_sim.cos_sim(ukr_sources, pure_disinfo_df)
ukr_disinfo_dist = cos_sim.cos_sim(ukr_sources, mixed_disinfo_df)



print(skrip_noninfl_dist)
print(ukr_noninfl_dist)




