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

dataset = 'skrip_v7' # options: skrip_v7, ukr_v3

if(dataset=='ukr_v3'):
    name = 'Ukraine'
if(dataset=='skrip_v7'):
    name = 'Skripal'

filename = f'results/RQ2_{name}_heatmap_df'
results_dir='results'

if os.path.exists(f'{filename}.pkl'):
    print("data found, loading...")
    #df_heatmap = pd.read_csv(f'{datapath}.csv', index_col=0)
    df_heatmap = pd.read_pkl(f'{filename}.pkl')
else:
    print("no data found, generating...")
    
    skrip_v7_te = 'data/Skripal/v7/indv_network/actor_te_edges_df.csv'
    ukr_v3_te = 'data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv'
    if dataset == 'skrip_v7':
        TE_df = pd.read_csv(skrip_v7_te)
    elif dataset == 'ukr_v3':
        TE_df = pd.read_csv(ukr_v3_te)
    edge_types = generate_edge_types.generate_edge_types()
    edge_types.remove('total_te')
    df_heatmap = pd.DataFrame(index= range(16), columns = edge_types)
    df_heatmap.index = edge_types
    for i in edge_types:
        for j in edge_types:
            df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
    df_heatmap.to_pickle(f'{filename}.pkl')
    df_heatmap.to_csv(f'{filename}.csv')

# drop columns if they are not an influence type:
print(df_heatmap)
df_heatmap = df_heatmap[df_heatmap.columns.intersection(edge_types)]
print(df_heatmap)
plot_heatmap.plot_layer_heatmap(df_heatmap, name, results_dir)


