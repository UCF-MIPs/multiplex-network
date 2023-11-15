import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from src import generate_edge_types
from src import add_aggregate_networks

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = generate_edge_types.generate_edge_types()
edge_types = edge_types + ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']

#dataset = 'ukr_v3' # options: skrip_v7, ukr_v3

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat','Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland','tanks','Zelensky_visit_the_US']

for dataset in dataset_list:
    # TODO fix? not sure if *_* == total_te
    if dataset!='Skripal':
        # rename total_te to *_*
        dict1 = {"total_te":"*_*"}
        edge_types = [dict1.get(n,n) for n in edge_types]

    # Data
    if dataset == 'Skripal':
        te_df_path = 'data/Skripal/v7/indv_network/actor_te_edges_df.csv'
    elif dataset == 'Ukraine':
        te_df_path = 'data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv'
    else:
        te_df_path = f'data/Scenarios/{dataset}_actor_te_edges_df.csv'

    # Networks
    graph_dict = {}
    edge_types2 = ['actors'] + edge_types

    out_infl_weights_df = pd.DataFrame(columns = edge_types2)
    in_infl_weights_df = pd.DataFrame(columns = edge_types2)

    # Pre-process #TODO fix to include chunking method
    graph_df = pd.read_csv(te_df_path)
    actors = pd.unique(graph_df[['Source', 'Target']].values.ravel('K'))
    out_infl_weights_df['actors'] = actors
    in_infl_weights_df['actors'] = actors

    # Check number of unique users
    num_us = len(pd.unique(graph_df[['Source', 'Target']].values.ravel('K')))
    print(f'number of unique users: {num_us}')

    graph_df = add_aggregate_networks.add_aggr_nets(graph_df)

    for edge_type in edge_types:
        graph_dict[edge_type] = {}
        
        g = nx.from_pandas_edgelist(graph_df, source='Source', target='Target', edge_attr=[edge_type], create_using=nx.DiGraph())
        #nx.relabel_nodes(g, actors, copy=False)

        graph_dict[edge_type] = g

        # identify which influence types nodes appear in, save summed weights
        for node in actors:
    
            ## out weight df filling
            if g.has_node(node):
                out_edges = g.out_edges(node, data=True)
                summed_weight = 0
                for edge_data in out_edges:
                    #convert 'dict_items' dtype to float
                    for k, v in edge_data[2].items():
                        w = float(v)
                    summed_weight += w
                row_index = out_infl_weights_df.index[out_infl_weights_df['actors']==node].to_list()
                out_infl_weights_df.loc[row_index, [edge_type]]=summed_weight

            ## in weight df filling
            if g.has_node(node):
                in_edges = g.in_edges(node, data=True)
                summed_weight = 0
                for edge_data in in_edges:
                    #iconvert 'dict_items' dtype to float
                    for k, v in edge_data[2].items():
                        w = float(v)
                    summed_weight += w
                row_index = in_infl_weights_df.index[in_infl_weights_df['actors']==node].to_list()
                in_infl_weights_df.loc[row_index, [edge_type]]=summed_weight

    out_infl_weights_df.to_csv(f'data/preprocessed/{dataset}_out_infl_weights_df.csv')
    in_infl_weights_df.to_csv(f'data/preprocessed/{dataset}_in_infl_weights_df.csv')

