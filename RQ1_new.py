import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src import plot_heatmap
from src import plot_participant_coef
from src import participation_coef as part_coef

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = ['T_T', 'U_U', 'T_U', 'U_T', 'TM_*', 'TF_*', 'UM_*', 'UF_*', 'T_*', 'U_*']
dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']
for dataset in dataset_list:
    results_dir_2TU = f'plots/{dataset}/TU'
    results_dir_4TUMF = f'plots/{dataset}/TUMF'

    ### Data ###
    out_infl_wdf = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_out_infl_weights_df.csv')

    ######## Heatmaps for each influence type #########

    ######## UM #########
    ### 4 Layers (TUMF) ###
    UM_out_aggr_wdf = out_infl_wdf[['actors', 'UM_*', 'UM_TM', 'UM_UM', 'UM_TF', 'UM_UF']]
    UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
    UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('UM', UM_out_aggr_wdf, dataset, results_dir_4TUMF, 'Blues')

    ######## TM #########
    ### 4 Layers (TUMF) ###
    TM_out_aggr_wdf = out_infl_wdf[['actors', 'TM_*', 'TM_TM', 'TM_UM', 'TM_TF', 'TM_UF']]
    TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
    TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('TM', TM_out_aggr_wdf, dataset, results_dir_4TUMF, 'Greens')

    ######## UF #########
    ### 4 Layers (TUMF) ###
    UF_out_aggr_wdf = out_infl_wdf[['actors', 'UF_*', 'UF_TM', 'UF_UM', 'UF_TF', 'UF_UF']]
    UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
    UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('UF', UF_out_aggr_wdf, dataset, results_dir_4TUMF, 'Reds')

    ######## TF #########
    ### 4 Layers (TUMF) ###
    TF_out_aggr_wdf = out_infl_wdf[['actors', 'TF_*', 'TF_TM', 'TF_UM', 'TF_TF', 'TF_UF']]
    TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
    TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_4layers('TF', TF_out_aggr_wdf, dataset, results_dir_4TUMF, 'Oranges')

    ######## Trustworthy #########
    ### 2 Layers (TU) ###
    T_out_aggr_wdf = out_infl_wdf[['actors', 'T_*', 'T_T', 'T_U']]
    T_out_aggr_wdf = T_out_aggr_wdf.loc[T_out_aggr_wdf['T_*'] !=0]
    T_out_aggr_wdf = T_out_aggr_wdf.sort_values(by=['T_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_2layers('T', T_out_aggr_wdf, dataset, results_dir_2TU, 'Purples')

    ######## Untrustworthy #########
    ### 2 Layers (TU) ###
    TF_out_aggr_wdf = out_infl_wdf[['actors', 'U_*', 'U_T', 'U_U']]
    TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['U_*'] !=0]
    TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['U_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree_2layers('U', TF_out_aggr_wdf, dataset, results_dir_2TU, 'Greys')

    #################################
    ### participation coefficient ###
    #################################

    # ## Creating Plots for Participant Coefficient of Sources ##
    # part_coef.part_coef_out(UM_out_aggr_wdf,'UM', 4)
    # part_coef.part_coef_out(TM_out_aggr_wdf,'TM', 4)
    # part_coef.part_coef_out(UF_out_aggr_wdf,'UF', 4)
    # part_coef.part_coef_out(TF_out_aggr_wdf,'TF', 4)
    # plt.clf()
    # plot_participant_coef.part_coef_plot_out('UM', UM_out_aggr_wdf, dataset, results_dir_out)
    # plt.clf()
    # plot_participant_coef.part_coef_plot_out('TM', TM_out_aggr_wdf, dataset, results_dir_out)
    # plt.clf()
    # plot_participant_coef.part_coef_plot_out('UF', UF_out_aggr_wdf, dataset, results_dir_out)
    # plt.clf()
    # plot_participant_coef.part_coef_plot_out('TF', TF_out_aggr_wdf, dataset, results_dir_out)



