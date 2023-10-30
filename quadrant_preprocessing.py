import pandas as pd
import os
from src import add_aggregate_networks
from src import layer_correlation

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']

for dataset in dataset_list:
    datapath = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing.csv'
    if os.path.exists(datapath):
        print("data has been found")
    else:
        print("data has not been found, generating ...")
        if dataset == 'Skripal':
            df = pd.read_csv('Data/raw/Skripal/Skripal_actor_te_edges_df.csv')
        elif dataset == 'Ukraine':
            df = pd.read_csv('Data/raw/Ukraine/Ukraine_actor_te_edges_df.csv')
        else:
            df = pd.read_csv(f'Data/raw/Scenarios/{dataset}_actor_te_edges_df.csv')
        aggregated_df = add_aggregate_networks.add_aggr_nets(df)
        aggregated_df.to_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing.csv')

### Quadrant crossover when nodes act as sources ###
for dataset in dataset_list:
    datapath = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources.csv'
    if os.path.exists(datapath):
        print("data has been found")
    else:
        print("data has not been found, generating ...")
        TE_df = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing.csv')
        multiplex_source = ['TM_*', 'TF_*', 'UM_*', 'UF_*']
        df_heatmap = pd.DataFrame(index= range(4), columns = multiplex_source)
        df_heatmap.index = multiplex_source
        for i in multiplex_source:
            for j in multiplex_source:
                df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
        df_heatmap.to_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources.csv')

### Quadrant crossover when nodes act as targets ###
for dataset in dataset_list:
    datapath = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_targets.csv'
    if os.path.exists(datapath):
        print("data has been found")
    else:
        print("data has not been found, generating ...")
        TE_df = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing.csv')
        multiplex_target = ['*_TM', '*_TF', '*_UM', '*_UF']
        df_heatmap = pd.DataFrame(index= range(4), columns = multiplex_target)
        df_heatmap.index = multiplex_target
        for i in multiplex_target:
            for j in multiplex_target:
                df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
        df_heatmap.to_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_targets.csv')