import networkx as nx
import pandas as pd
import numpy as np
from src import generate_edge_types

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = generate_edge_types.generate_edge_types()

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']

for dataset in dataset_list:
    if dataset != 'Skripal':
        # rename total_te to *_*
        dict1 = {"total_te":"*_*"}
        edge_types = [dict1.get(n,n) for n in edge_types]

    # Data
    if dataset == 'Skripal':
        te_df_path = 'Data/raw/Skripal/Skripal_actor_te_edges_df.csv'
    elif dataset == 'Ukraine':
        te_df_path = 'Data/raw/Ukraine/Ukraine_actor_te_edges_df.csv'
    else:
        te_df_path = f'Data/raw/Scenarios/{dataset}_actor_te_edges_df.csv'
    graph_df = pd.read_csv(te_df_path)
    print('Sources', graph_df['Source'].nunique())
    print('Targets', graph_df['Target'].nunique())
    print(np.count_nonzero(graph_df['*_*']))
