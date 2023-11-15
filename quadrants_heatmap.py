import pandas as pd
import numpy as np
import os
from src import plot_heatmap
from src import add_aggregate_networks 
from src import layer_correlation
from matplotlib import pyplot as plt

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat', 'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland', 'tanks','Zelensky_visit_the_US']


for dataset in dataset_list:
    if dataset == 'Skripal':
        df = pd.read_csv('data/Skripal/v7/indv_network/actor_te_edges_df.csv')
    elif dataset == 'Ukraine':
        df = pd.read_csv('data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv')
    else:
        df = pd.read_csv(f'data/scenarios/{dataset}_actor_te_edges_df.csv')
    TE_df = add_aggregate_networks.add_aggr_nets(df)
    #aggregated_df.to_csv(f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing.csv')
    ### Quadrant crossover when nodes act as sources ###
    multiplex_source = ['TM_*', 'TF_*', 'UM_*', 'UF_*']
    df_heatmap = pd.DataFrame(index= range(4), columns = multiplex_source)
    df_heatmap.index = multiplex_source
    for i in multiplex_source:
        for j in multiplex_source:
            df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
    df_heatmap.to_csv(f'data/preprocessed/{dataset}_quadrant_preprocessing_sources.csv')
    df_heatmap_source = df_heatmap
    plot_heatmap.plot_layer_heatmap(df_heatmap_source, f'{dataset}_quadrant', f'results/plots')
    plt.clf()

