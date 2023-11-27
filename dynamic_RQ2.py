import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import glob
from pathlib import Path
from src import plot_heatmap
from src import layer_correlation
from src import add_aggregate_networks
from src import generate_edge_types
from src import cos_sim

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = generate_edge_types.generate_edge_types()
edge_types = edge_types + ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
edge_types.remove('total_te')

dataset_list = ['Mariupol_hospital']
#dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat','Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland','tanks','Zelensky_visit_the_US']


# template matrices
pure_disinfo = np.array([1.0, 0.0, 0.0, 0.0, \
                           0.0, 1.0, 0.0, 0.0, \
                           1.0, 1.0, 1.0, 0.0, \
                           1.0, 1.0, 0.0, 1.0])

for dataset in dataset_list: 
 
    # Data
    #if dataset == 'Skripal':
        #...
    #elif dataset == 'Ukraine':
        #...
    #else:
    csvs = glob.glob(f'data/dynamic/Mariupol_hospital/*.csv')   
    
    #Resulting array
    shape = (len(csvs),4,4) # time, 2d array
    heatmaps = np.zeros(shape)
    sim_to_disinfl = np.zeros(len(csvs)) # 1d comparison of TE to disinfluence template

    for t, csvfile in enumerate(csvs):
        name = csvfile.replace(str('data/dynamic/'+dataset + "/actor_te_edges_df"), dataset)
        name = name.replace('.csv', '_preprocessed')
        datapath = f'results/dynamic/Scenarios/{dataset}/'
        
        ### PREPROCESSING ###
        if os.path.exists(f'{datapath}{name}.csv'):
            print(f"{dataset} preprocessed data found, loading...")
            df_heatmap = pd.read_csv(f'{datapath}{name}.csv', index_col=0)
            #df_heatmap = pd.read_pkl(f'{filename}.pkl')
        else:
            print(f"no preprocessed data found for {dataset}, generating...") 

            # Networks
            graph_dict = {}
            edge_types2 = ['actors'] + edge_types

            out_infl_weights_df = pd.DataFrame(columns = edge_types2)
            in_infl_weights_df = pd.DataFrame(columns = edge_types2)

            #TODO fix to include chunking method
            graph_df = pd.read_csv(csvfile)
            actors = pd.unique(graph_df[['Source', 'Target']].values.ravel('K'))
            out_infl_weights_df['actors'] = actors
            in_infl_weights_df['actors'] = actors

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

            out_infl_weights_df.to_csv(f'{datapath}{name}.csv')
     
        ### 16x16 TE HEATMAPS ###
        name = csvfile.replace(str('data/dynamic/'+dataset + "/actor_te_edges_df"), dataset)
        name = name.replace('.csv', '_heatmap_df')

        if os.path.exists(f'{datapath}{name}.csv'):
            print(f"{dataset} heatmap data found, loading...")
            df_heatmap = pd.read_csv(f'{datapath}{name}.csv', index_col=0)
            #df_heatmap = pd.read_pkl(f'{filename}.pkl')
        else:
            print(f"no heatmap data found for {dataset}, generating...") 
            TE_df = pd.read_csv(csvfile) #TODO update to pkl
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
            df_heatmap.to_csv(f'{datapath}{name}.csv')

        ### 4x4 TE HEATMAPS ###
        #if os.path.exists(f'{datapath}dynamic_heatmaps.npy'):
        #    pass
        #else:
        df = pd.read_csv(csvfile)
        TE_df = add_aggregate_networks.add_aggr_nets(df)
        multiplex_source = ['TM_*', 'TF_*', 'UM_*', 'UF_*']
        df_heatmap = pd.DataFrame(index= range(4), columns = multiplex_source)
        df_heatmap.index = multiplex_source
        for i in multiplex_source:
            for j in multiplex_source:
                df_heatmap.at[i, j] = layer_correlation.layer_correlation(TE_df, i, j)
        heatmaps[t]=df_heatmap.to_numpy()
        sim_to_disinfl[t] = cos_sim.cos_sim(pure_disinfo, heatmaps[t])
        print(f'{heatmaps[t]}')
        print(f' cosine similarity to disinfo at time {t}: {sim_to_disinfl[t]}')
    # visuals
    np.save(f'{datapath}dynamic_heatmaps.npy', heatmaps)
    time = np.arange(13)
    print(time)
    print(sim_to_disinfl)
    print(np.size(time), np.size(sim_to_disinfl))
    plt.scatter(time, sim_to_disinfl)
    plt.title(f'{dataset} disinfluence indicator')
    plt.xlabel('time (3days)')
    plt.ylabel('cos similarity to disinfluence template')
    plt.savefig(f'{datapath}dynamic.png')
    plt.clf()

