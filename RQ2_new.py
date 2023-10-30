import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from src import generate_edge_types
from src import plot_heatmap
from src import layer_correlation
from src import generate_edge_types

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']

for dataset in dataset_list:
    # Data
    if dataset == 'Skripal':
        te_df_path = 'Data/raw/Skripal/Skripal_actor_te_edges_df.csv'
    elif dataset == 'Ukraine':
        te_df_path = 'Data/raw/Ukraine/Ukraine_actor_te_edges_df.csv'
    else:
        te_df_path = f'Data/raw/Scenarios/{dataset}_actor_te_edges_df.csv'

    datapath = f'results/RQ2_{dataset}_heatmap_df.csv'
    results_dir = f'plots/{dataset}'

    if os.path.exists(datapath):
        print("data found, loading...")
        df_heatmap = pd.read_csv(datapath, index_col=0)
    else:
        print("no data found, generating...")

        TE_df = pd.read_csv(te_df_path)
        edge_types = generate_edge_types.generate_edge_types()
        edge_types.remove('total_te')
        df_heatmap = pd.DataFrame(index= range(16), columns = edge_types)
        df_heatmap.index = edge_types
        for i in edge_types:
            for j in edge_types:
                df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)

        df_heatmap.to_csv(f'results/RQ2_{dataset}_heatmap_df.csv')

    # drop columns if they are not an influence type:
    print(df_heatmap)
    df_heatmap = df_heatmap[df_heatmap.columns.intersection(edge_types)]
    print(df_heatmap)
    plot_heatmap.plot_layer_heatmap(df_heatmap, dataset, results_dir)


