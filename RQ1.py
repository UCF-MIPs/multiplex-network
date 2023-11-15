#TODO
# get ukr/skrip working simultaneously in node_correlation
# put participation coeff calculations in loops

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src import plot_heatmap
from src import plot_participant_coef
from src import participation_coef as part_coef

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
    
edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat', 'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland', 'tanks','Zelensky_visit_the_US']


for dataset in dataset_list:
    #results_dir_in = f'results/{dataset}/Targets'
    results_dir_out = f'results/RQ1_Sources'
    
    #in_infl_wdf = pd.read_csv(f'data/preprocessed/{dataset}_in_infl_weights_df.csv')
    out_infl_wdf = pd.read_csv(f'data/preprocessed/{dataset}_out_infl_weights_df.csv')

    ######## Heatmaps for each influence type #########

    ######## UM #########
    ## Sources/ Outgoing ##
    UM_out_aggr_wdf = out_infl_wdf[['actors', 'UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF']]
    UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
    UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('UM', UM_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    #UM_in_aggr_wdf = in_infl_wdf[['actors', '*_UM', 'UM_UM', 'TM_UM', 'UF_UM', 'TF_UM']]
    #UM_in_aggr_wdf = UM_in_aggr_wdf.loc[UM_in_aggr_wdf['*_UM'] !=0]
    #UM_in_aggr_wdf = UM_in_aggr_wdf.sort_values(by=['*_UM'], ascending=False).dropna()
    #plot_heatmap.plot_user_heatmap_indegree('UM', UM_in_aggr_wdf, name, results_dir_in)



    ######## TM #########
    ## Sources/ Outgoing ##
    TM_out_aggr_wdf = out_infl_wdf[['actors', 'TM_*', 'TM_TM', 'TM_UM', 'TM_TF', 'TM_UF']]
    TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
    TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('TM', TM_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    #TM_in_aggr_wdf = in_infl_wdf[['actors', '*_TM', 'TM_TM', 'UM_TM', 'TF_TM', 'UF_TM']]
    #TM_in_aggr_wdf = TM_in_aggr_wdf.loc[TM_in_aggr_wdf['*_TM'] !=0]
    #TM_in_aggr_wdf = TM_in_aggr_wdf.sort_values(by=['*_TM'], ascending=False).dropna()
    #plot_heatmap.plot_user_heatmap_indegree('TM', TM_in_aggr_wdf, name, results_dir_in)


    ######## UF #########
    ## Sources/ Outgoing ##
    UF_out_aggr_wdf = out_infl_wdf[['actors', 'UF_*', 'UF_UF', 'UF_TF', 'UF_UM', 'UF_TM']]
    UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
    UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('UF', UF_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    #UF_in_aggr_wdf = in_infl_wdf[['actors', '*_UF', 'UF_UF', 'TF_UF', 'UM_UF', 'TM_UF']]
    #UF_in_aggr_wdf = UF_in_aggr_wdf.loc[UF_in_aggr_wdf['*_UF'] !=0]
    #UF_in_aggr_wdf = UF_in_aggr_wdf.sort_values(by=['*_UF'], ascending=False).dropna()
    #plot_heatmap.plot_user_heatmap_indegree('UF', UF_in_aggr_wdf, name, results_dir_in)



    ######## TF #########
    ## Sources/ Outgoing ##
    TF_out_aggr_wdf = out_infl_wdf[['actors', 'TF_*', 'TF_TF', 'TF_UF', 'TF_TM', 'TF_UM']]
    TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
    TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False).dropna()
    plot_heatmap.plot_user_heatmap_outdegree('TF', TF_out_aggr_wdf, dataset, results_dir_out)

    ## Targets/ Incoming ##
    #TF_in_aggr_wdf = in_infl_wdf[['actors', '*_TF', 'TF_TF', 'UF_TF', 'TM_TF', 'UM_TF']]
    #TF_in_aggr_wdf = TF_in_aggr_wdf.loc[TF_in_aggr_wdf['*_TF'] !=0]
    #TF_in_aggr_wdf = TF_in_aggr_wdf.sort_values(by=['*_TF'], ascending=False).dropna()
    #plot_heatmap.plot_user_heatmap_indegree('TF', TF_in_aggr_wdf, name, results_dir_in)

    #################################
    ### participation coefficient ###
    #################################

    part_cof_UM = part_coef.part_coef(out_infl_wdf, 'UM')
    part_cof_TM = part_coef.part_coef(out_infl_wdf, 'TM')
    part_cof_UF = part_coef.part_coef(out_infl_wdf, 'UF')
    part_cof_TF = part_coef.part_coef(out_infl_wdf, 'TF')

    part_df = pd.DataFrame({'actors': out_infl_wdf['actors'],\
                            'p_UM': part_cof_UM,\
                            'p_TM': part_cof_TM,\
                            'p_UF': part_cof_UF,\
                            'p_TF': part_cof_TF})
    
    part_df = part_df.dropna(how='all')
    part_df = part_df[ \
                part_df['p_UM'].notna() | \
                part_df['p_TM'].notna() | \
                part_df['p_UF'].notna() | \
                part_df['p_TF'].notna() \
                ]

    part_df.to_csv(f'results/{dataset}_out_part_coef.csv')

    ## Creating Plots for Participant Coefficient of Sources ##
    UM_out = part_coef.part_coef_out(UM_out_aggr_wdf,'UM', 4)
    TM_out = part_coef.part_coef_out(TM_out_aggr_wdf,'TM', 4)
    UF_out = part_coef.part_coef_out(UF_out_aggr_wdf,'UF', 4)
    TF_out = part_coef.part_coef_out(TF_out_aggr_wdf,'TF', 4)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('UM', UM_out, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('TM', TM_out, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('UF', UF_out, dataset, results_dir_out)
    plt.clf()
    plot_participant_coef.part_coef_plot_out('TF', TF_out, dataset, results_dir_out)



### COMPARING ALL AGGREGATES
#TODO finish this
'''
# Shows comparitive heatmap of activity of influence sources vs. audience

#in_aggr_infl_wdf = in_infl_wdf[['UM_*', 'UF_*', 'TM_*', 'TF_*']]
#out_aggr_infl_wdf = out_infl_wdf[['UM_*', 'UF_*', 'TM_*', 'TF_*']]

# To include plot of corresponding index of in to out plot and out to in plot:
#https://stackoverflow.com/questions/58423707/pandas-matching-index-of-one-dataframe-to-the-column-of-other-dataframe
# Would show that outgoing actors are/are not the same as incoming

# sort by value of first column

out_aggr_infl_weights_df = out_aggr_infl_wdf.sort_values(by=['UM_*'], ascending=False)
in_aggr_infl_weights_df = in_aggr_infl_wdf.sort_values(by=['UM_*'], ascending=False)

#inout_df = in_aggr_infl_weights_df['actors']
#pd.merge(inout_df[['actors']], out_aggr_infl_weights_df, how='left', on='actors', sort=False)
#print(inout_df)
#outin_df = in_infl_weights_df[edge_types].reindex(out_aggr_infl_weights_df.index)

# incoming edges plot
heatmap = np.empty((4, len(in_aggr_infl_weights_df['UM_*'])))

for i, (column_name, column_data) in enumerate(in_aggr_infl_weights_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data
        pass

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title(f'{dataset} target actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_aggr_in_node_activity.png')

# corresponding order for out
heatmap = np.empty((4, 2001))

for i, (column_name, column_data) in enumerate(inout_df.items()):
    print(column_name)
    print(column_data)

    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title('target actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig('in2out_node_activity.png')

# outgoing edges plot
heatmap = np.empty((4, len(out_aggr_infl_weights_df['UM_*'])))

for i, (column_name, column_data) in enumerate(out_aggr_infl_weights_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title(f'{dataset} source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_aggr_out_node_activity.png')

# corresponding order for in 
heatmap = np.empty((4, 2001))

for i, (column_name, column_data) in enumerate(outin_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title('source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig('out2in_node_activity.png')

'''

