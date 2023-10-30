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

dataset = 'skrip_v7' # options: skrip_v7, ukr_v3
# Data
skrip_v7_te = 'Data/raw/Skripal/actor_te_edges_df - Skripal.csv'
ukr_v3_te = 'Data/raw/Ukraine/actor_te_edges_df_2022_01_01_2022_05_01 - Ukraine.csv'

if(dataset=='ukr_v3'):
    name = 'Ukraine'
if(dataset=='skrip_v7'):
    name = 'Skripal'

datapath = f'results/RQ2_{name}_heatmap_df.csv'
results_dir = f'plots/{name}/RQ2'

if os.path.exists(datapath):
    print("data found, loading...")
    df_heatmap = pd.read_csv(datapath, index_col=0)
else:
    print("no data found, generating...")

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
            df_heatmap.at[i, j] = layer_correlation(TE_df, i, j)

    df_heatmap.to_csv(f'results/RQ2_{name}_heatmap_df.csv')

# drop columns if they are not an influence type:
print(df_heatmap)
df_heatmap = df_heatmap[df_heatmap.columns.intersection(edge_types)]
print(df_heatmap)
plot_heatmap.plot_layer_heatmap(df_heatmap, name, results_dir)


