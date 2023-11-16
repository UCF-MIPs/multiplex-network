import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from src import generate_edge_types
from src import plot_heatmap
from src import layer_correlation

edge_types = generate_edge_types.generate_edge_types()
edge_types = edge_types + ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
#dataset = 'skrip_v7' # options: skrip_v7, ukr_v3
dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat', 'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland', 'tanks','Zelensky_visit_the_US']

for dataset in dataset_list:
    # Data
    if dataset == 'Skripal':
        te_df_path = 'data/Skripal/v7/indv_network/actor_te_edges_df.csv'
    elif dataset == 'Ukraine':
        te_df_path = 'data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv'
    else:
        te_df_path = f'data/scenarios/{dataset}_actor_te_edges_df.csv'

    datapath = f'results/RQ2/{dataset}_heatmap_df'
    results_dir='results/RQ2'

    if os.path.exists(f'{datapath}.csv'):
        print(f"{dataset} data found, loading...")
        df_heatmap = pd.read_csv(f'{datapath}.csv', index_col=0)
        #df_heatmap = pd.read_pkl(f'{filename}.pkl')
    else:
        print(f"no data found for {dataset}, generating...") 
        TE_df = pd.read_csv(te_df_path) #TODO update to pkl
        # remove empty rows, was getting div by 0 error on Mariupol theater data
        Te_df = TE_df.dropna(how='all')
        #didn't do it, #TODO fix this
        edge_types = generate_edge_types.generate_edge_types()
        edge_types.remove('total_te')
        df_heatmap = pd.DataFrame(index= range(16), columns = edge_types)
        df_heatmap.index = edge_types
        for i in edge_types:
            for j in edge_types:
                df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
        #df_heatmap.to_pickle(f'{datapath}.pkl')
        df_heatmap.to_csv(f'{datapath}.csv')

    # drop columns if they are not an influence type:
    print(f'plotting {dataset}')
    #print(df_heatmap)
    df_heatmap = df_heatmap[df_heatmap.columns.intersection(edge_types)]
    #print(df_heatmap)
    plot_heatmap.plot_layer_heatmap(df_heatmap, dataset, results_dir)
    plt.clf()

