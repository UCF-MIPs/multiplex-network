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

edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']
for dataset in dataset_list:
    results_dir_in = f'plots/{dataset}/Targets'
    results_dir_out = f'plots/{dataset}/Sources'

    in_infl_wdf = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_in_infl_weights_df.csv')
    out_infl_wdf = pd.read_csv(f'Data/preprocessed/{dataset}/{dataset}_out_infl_weights_df.csv')

    ######## Heatmaps for each influence type #########

    ######## UM #########
    ## Sources/ Outgoing ##
    UM_out_aggr_wdf = out_infl_wdf[['actors', 'UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF']]
    UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
    UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('UM', UM_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    UM_in_aggr_wdf = in_infl_wdf[['actors', '*_UM', 'UM_UM', 'TM_UM', 'UF_UM', 'TF_UM']]
    UM_in_aggr_wdf = UM_in_aggr_wdf.loc[UM_in_aggr_wdf['*_UM'] !=0]
    UM_in_aggr_wdf = UM_in_aggr_wdf.sort_values(by=['*_UM'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_indegree('UM', UM_in_aggr_wdf, dataset, results_dir_in)


    ######## TM #########
    ## Sources/ Outgoing ##
    TM_out_aggr_wdf = out_infl_wdf[['actors', 'TM_*', 'TM_TM', 'TM_UM', 'TM_TF', 'TM_UF']]
    TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
    TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('TM', TM_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    TM_in_aggr_wdf = in_infl_wdf[['actors', '*_TM', 'TM_TM', 'UM_TM', 'TF_TM', 'UF_TM']]
    TM_in_aggr_wdf = TM_in_aggr_wdf.loc[TM_in_aggr_wdf['*_TM'] !=0]
    TM_in_aggr_wdf = TM_in_aggr_wdf.sort_values(by=['*_TM'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_indegree('TM', TM_in_aggr_wdf, dataset, results_dir_in)


    ######## UF #########
    ## Sources/ Outgoing ##
    UF_out_aggr_wdf = out_infl_wdf[['actors', 'UF_*', 'UF_UF', 'UF_TF', 'UF_UM', 'UF_TM']]
    UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
    UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('UF', UF_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    UF_in_aggr_wdf = in_infl_wdf[['actors', '*_UF', 'UF_UF', 'TF_UF', 'UM_UF', 'TM_UF']]
    UF_in_aggr_wdf = UF_in_aggr_wdf.loc[UF_in_aggr_wdf['*_UF'] !=0]
    UF_in_aggr_wdf = UF_in_aggr_wdf.sort_values(by=['*_UF'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_indegree('UF', UF_in_aggr_wdf, dataset, results_dir_in)


    ######## TF #########
    ## Sources/ Outgoing ##
    TF_out_aggr_wdf = out_infl_wdf[['actors', 'TF_*', 'TF_TF', 'TF_UF', 'TF_TM', 'TF_UM']]
    TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
    TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('TF', TF_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    TF_in_aggr_wdf = in_infl_wdf[['actors', '*_TF', 'TF_TF', 'UF_TF', 'TM_TF', 'UM_TF']]
    TF_in_aggr_wdf = TF_in_aggr_wdf.loc[TF_in_aggr_wdf['*_TF'] !=0]
    TF_in_aggr_wdf = TF_in_aggr_wdf.sort_values(by=['*_TF'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_indegree('TF', TF_in_aggr_wdf, dataset, results_dir_in)


    #################################
    ### participation coefficient ###
    #################################

    ### new participant coefficient functions ###
    ## Sources/outdegree ##
    def part_coef_out(df, inf, layers):
        types = ['UM','TM','UF','TF']
        aggr_type = str(inf + '_*')
        df['ptemp'] = 0
        for i in types:
            infl_type = str(inf + '_' + i)
            df['ptemp'] += np.power(np.divide(df[infl_type],df[aggr_type]),2)
        df['pc_out'] = (layers/(layers-1))*(1-df['ptemp'])
        return df 

    ## Targets/Incoming ##
    def part_coef_in(df, inf, layers):
        types = ['UM','TM','UF','TF']
        aggr_type = str('*_' + inf)
        df['ptemp'] = 0
        for i in types:
            infl_type = str(i + '_' + inf)
            df['ptemp'] += np.power(np.divide(df[infl_type],df[aggr_type]),2)
        df['pc_in'] = (layers/(layers-1))*(1-df['ptemp'])
        return df 

    ## Creating Plots for Participant Coefficient of Sources ##
    part_coef.part_coef_out(UM_out_aggr_wdf,'UM', 4)
    part_coef.part_coef_out(TM_out_aggr_wdf,'TM', 4)
    part_coef.part_coef_out(UF_out_aggr_wdf,'UF', 4)
    part_coef.part_coef_out(TF_out_aggr_wdf,'TF', 4)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('UM', UM_out_aggr_wdf, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('TM', TM_out_aggr_wdf, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('UF', UF_out_aggr_wdf, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('TF', TF_out_aggr_wdf, dataset, results_dir_out)

    ## Creating Plots for Participant Coefficient of Targets ##
    part_coef.part_coef_in(UM_in_aggr_wdf,'UM', 4)
    part_coef.part_coef_in(TM_in_aggr_wdf,'TM', 4)
    part_coef.part_coef_in(UF_in_aggr_wdf,'UF', 4)
    part_coef.part_coef_in(TF_in_aggr_wdf,'TF', 4)
    plt.clf()
    plot_participant_coef.part_coef_plot_in('UM', UM_in_aggr_wdf, dataset, results_dir_in)
    plt.clf()
    plot_participant_coef.part_coef_plot_in('TM', TM_in_aggr_wdf, dataset, results_dir_in)
    plt.clf()
    plot_participant_coef.part_coef_plot_in('UF', UF_in_aggr_wdf, dataset, results_dir_in)
    plt.clf()
    plot_participant_coef.part_coef_plot_in('TF', TF_in_aggr_wdf, dataset, results_dir_in)

