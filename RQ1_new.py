import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import generate_edge_types
from src import plot_heatmap
from src import plot_participant_coef
from src import participation_coef as part_coef

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*', 'T_*', 'U_*']

accumulated_UM_4layer_df = pd.DataFrame()
accumulated_TM_4layer_df = pd.DataFrame()
accumulated_UF_4layer_df = pd.DataFrame()
accumulated_TF_4layer_df = pd.DataFrame()
accumulated_T_2layer_df = pd.DataFrame()
accumulated_U_2layer_df = pd.DataFrame()

for dataset in generate_edge_types.dataset_list:
    results_dir_2TU = f'plots/{dataset}/TU'
    results_dir_4TUMF = f'plots/{dataset}/TUMF'

    ############
    ### Data ###
    ############
    out_infl_wdf = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_out_infl_weights_df.csv')

    ###################################################
    ######## Heatmaps for each influence type #########
    ###################################################

    ######## UM #########
    ### 4 Layers (TUMF) ###
    UM_out_aggr_wdf = out_infl_wdf[['actors', 'UM_*', 'UM_TM', 'UM_UM', 'UM_TF', 'UM_UF']]
    UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
    UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('UM', UM_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF, 'BrBG')
    
    ######## TM #########
    ### 4 Layers (TUMF) ###
    TM_out_aggr_wdf = out_infl_wdf[['actors', 'TM_*', 'TM_TM', 'TM_UM', 'TM_TF', 'TM_UF']]
    TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
    TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('TM', TM_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF, 'RdBu')

    ######## UF #########
    ### 4 Layers (TUMF) ###
    UF_out_aggr_wdf = out_infl_wdf[['actors', 'UF_*', 'UF_TM', 'UF_UM', 'UF_TF', 'UF_UF']]
    UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
    UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('UF', UF_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF, 'PiYG')

    ######## TF #########
    ### 4 Layers (TUMF) ###
    TF_out_aggr_wdf = out_infl_wdf[['actors', 'TF_*', 'TF_TM', 'TF_UM', 'TF_TF', 'TF_UF']]
    TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
    TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('TF', TF_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF, 'PuOr')

    ######## Trustworthy #########
    ### 2 Layers (TU) ###
    T_out_aggr_wdf = out_infl_wdf[['actors', 'T_*', 'T_T', 'T_U']]
    T_out_aggr_wdf = T_out_aggr_wdf.loc[T_out_aggr_wdf['T_*'] !=0]
    T_out_aggr_wdf = T_out_aggr_wdf.sort_values(by=['T_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_2layers('T', T_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_2TU, 'RdBu')

    ######## Untrustworthy #########
    ### 2 Layers (TU) ###
    U_out_aggr_wdf = out_infl_wdf[['actors', 'U_*', 'U_T', 'U_U']]
    U_out_aggr_wdf = U_out_aggr_wdf.loc[U_out_aggr_wdf['U_*'] !=0]
    U_out_aggr_wdf = U_out_aggr_wdf.sort_values(by=['U_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_2layers('U', U_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_2TU, 'RdBu')

    #################################
    ### participation coefficient ###
    #################################

    ## Creating Plots for Participant Coefficient of Sources ##
    part_coef.part_coef_out_4layers(UM_out_aggr_wdf,'UM', 4)
    part_coef.part_coef_out_4layers(TM_out_aggr_wdf,'TM', 4)
    part_coef.part_coef_out_4layers(UF_out_aggr_wdf,'UF', 4)
    part_coef.part_coef_out_4layers(TF_out_aggr_wdf,'TF', 4)
    part_coef.part_coef_out_2layers(T_out_aggr_wdf,'T', 2)
    part_coef.part_coef_out_2layers(U_out_aggr_wdf,'U', 2)
    plt.clf()
    plot_participant_coef.part_coef_plot_out_4layers('UM', UM_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF,'#964B00','#005249')
    plt.clf()
    plot_participant_coef.part_coef_plot_out_4layers('TM', TM_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF,'#0000FF', '#FF0000')
    plt.clf()
    plot_participant_coef.part_coef_plot_out_4layers('UF', UF_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF,'#DB39d9', '#00FF00')
    plt.clf()
    plot_participant_coef.part_coef_plot_out_4layers('TF', TF_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_4TUMF,'#800080', '#DBA830')
    plt.clf()
    plot_participant_coef.part_coef_plot_out_2layers('T', T_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_2TU)
    plt.clf()
    plot_participant_coef.part_coef_plot_out_2layers('U', U_out_aggr_wdf, dataset, generate_edge_types.dict_list, results_dir_2TU)

    ######## Accumulated Dataframes #########
    ######### UM #########
    UM_out_aggr_wdf.rename(columns={'UM_*':f'UM_*_{dataset}'}, inplace=True)
    accumulated_UM_4layer_df = pd.concat([accumulated_UM_4layer_df, UM_out_aggr_wdf[f'UM_*_{dataset}']], axis=1, ignore_index=True)

    ######### TM #########
    TM_out_aggr_wdf.rename(columns={'TM_*':f'TM_*_{dataset}'}, inplace=True)
    accumulated_TM_4layer_df = pd.concat([accumulated_TM_4layer_df, TM_out_aggr_wdf[f'TM_*_{dataset}']], axis=1, ignore_index=True)

    ######### UF #########
    UF_out_aggr_wdf.rename(columns={'UF_*':f'UF_*_{dataset}'}, inplace=True)
    accumulated_UF_4layer_df = pd.concat([accumulated_UF_4layer_df, UF_out_aggr_wdf[f'UF_*_{dataset}']], axis=1, ignore_index=True)

    ######### TF #########
    TF_out_aggr_wdf.rename(columns={'TF_*':f'TF_*_{dataset}'}, inplace=True)
    accumulated_TF_4layer_df = pd.concat([accumulated_TF_4layer_df, TF_out_aggr_wdf[f'TF_*_{dataset}']], axis=1, ignore_index=True)

    ######### Trustworthy #########
    T_out_aggr_wdf.rename(columns={'T_*':f'T_*_{dataset}'}, inplace=True)
    accumulated_T_2layer_df = pd.concat([accumulated_T_2layer_df, T_out_aggr_wdf[f'T_*_{dataset}']], axis=1, ignore_index=True)

    ######### Untrustworthy #########
    U_out_aggr_wdf.rename(columns={'U_*':f'U_*_{dataset}'}, inplace=True)
    accumulated_U_2layer_df = pd.concat([accumulated_U_2layer_df, U_out_aggr_wdf[f'U_*_{dataset}']], axis=1, ignore_index=True)

columns_names = {0: 'Skripal', 1: 'Ukraine', 2: 'Navalny'}

######### UM #########
accumulated_UM_4layer_df.fillna(0, inplace=True)
accumulated_UM_4layer_df = pd.DataFrame(np.sort(accumulated_UM_4layer_df.values, axis=0)[::-1])
accumulated_UM_4layer_df.rename(columns=columns_names, inplace=True)
accumulated_UM_4layer_df = accumulated_UM_4layer_df.loc[(accumulated_UM_4layer_df!=0).any(axis=1)]
accumulated_UM_4layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_UM.csv')
print(accumulated_UM_4layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('UM', accumulated_UM_4layer_df, 'plots/Accumulated', 'RdBu')

######### TM #########
accumulated_TM_4layer_df.fillna(0, inplace=True)
accumulated_TM_4layer_df = pd.DataFrame(np.sort(accumulated_TM_4layer_df.values, axis=0)[::-1])
accumulated_TM_4layer_df.rename(columns=columns_names, inplace=True)
accumulated_TM_4layer_df = accumulated_TM_4layer_df.loc[(accumulated_TM_4layer_df!=0).any(axis=1)]
accumulated_TM_4layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_TM.csv')
print(accumulated_TM_4layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('TM', accumulated_TM_4layer_df, 'plots/Accumulated', 'RdBu')

######### UF #########
accumulated_UF_4layer_df.fillna(0, inplace=True)
accumulated_UF_4layer_df = pd.DataFrame(np.sort(accumulated_UF_4layer_df.values, axis=0)[::-1])
accumulated_UF_4layer_df.rename(columns=columns_names, inplace=True)
accumulated_UF_4layer_df = accumulated_UF_4layer_df.loc[(accumulated_UF_4layer_df!=0).any(axis=1)]
accumulated_UF_4layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_UF.csv')
print(accumulated_UF_4layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('UF', accumulated_UF_4layer_df, 'plots/Accumulated', 'RdBu')

######### TF #########
accumulated_TF_4layer_df.fillna(0, inplace=True)
accumulated_TF_4layer_df = pd.DataFrame(np.sort(accumulated_TF_4layer_df.values, axis=0)[::-1])
accumulated_TF_4layer_df.rename(columns=columns_names, inplace=True)
accumulated_TF_4layer_df = accumulated_TF_4layer_df.loc[(accumulated_TF_4layer_df!=0).any(axis=1)]
accumulated_TF_4layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_TF.csv')
print(accumulated_TF_4layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('TF', accumulated_TF_4layer_df, 'plots/Accumulated', 'RdBu')

######### Trustworthy #########
accumulated_T_2layer_df.fillna(0, inplace=True)
accumulated_T_2layer_df = pd.DataFrame(np.sort(accumulated_T_2layer_df.values, axis=0)[::-1])
accumulated_T_2layer_df.rename(columns=columns_names, inplace=True)
accumulated_T_2layer_df = accumulated_T_2layer_df.loc[(accumulated_T_2layer_df!=0).any(axis=1)]
accumulated_T_2layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_T.csv')
print(accumulated_T_2layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('T', accumulated_T_2layer_df, 'plots/Accumulated', 'RdBu')

######### Untrustworthy #########
accumulated_U_2layer_df.fillna(0, inplace=True)
accumulated_U_2layer_df = pd.DataFrame(np.sort(accumulated_U_2layer_df.values, axis=0)[::-1])
accumulated_U_2layer_df.rename(columns=columns_names, inplace=True)
accumulated_U_2layer_df = accumulated_U_2layer_df.loc[(accumulated_U_2layer_df!=0).any(axis=1)]
accumulated_U_2layer_df.to_csv('Data/preprocessed/Accumulated/accumulated_U.csv')
print(accumulated_U_2layer_df.shape)
plot_heatmap.plot_user_heatmap_datasets_comparison('U', accumulated_U_2layer_df, 'plots/Accumulated', 'RdBu')
