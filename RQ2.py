import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src import generate_edge_types

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

dataset = 'skrip_v7' # options: skrip_v7, ukr_v3
# Data
skrip_v7_te = 'Data/raw/Skripal/actor_te_edges_df - Skripal.csv'
ukr_v3_te = 'Data/raw/Ukraine/actor_te_edges_df_2022_01_01_2022_05_01 - Ukraine.csv'

if dataset == 'skrip_v7':
    TE_df = pd.read_csv(skrip_v7_te)
elif dataset == 'ukr_v3':
    TE_df = pd.read_csv(ukr_v3_te)

def layer_correlation(df, source, target):
    sum_of_numerator = 0
    sum_of_denominator = 0
    for index, row in df.iterrows():
        if row[source] > 0 and row[target] > 0:
            sum_of_numerator += 1
        else:
            sum_of_numerator += 0
    
    for index, row in df.iterrows():
        if row[target] > 0:
            sum_of_denominator += 1
        else:
            sum_of_denominator += 0
    
    return sum_of_numerator / sum_of_denominator

edge_types = generate_edge_types.generate_edge_types()
edge_types.remove('total_te')

df_heatmap = pd.DataFrame(index= range(16), columns = edge_types)
df_heatmap.index = edge_types
for i in edge_types:
    for j in edge_types:
        df_heatmap.at[i, j] = layer_correlation(TE_df, i, j)

df_heatmap.to_csv(f'results/RQ2_heatmap_df.csv')