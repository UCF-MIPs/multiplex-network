import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################################################
### creating plots for the same sources (4 Layers) ###
######################################################

def plot_user_heatmap_outdegree_4layers(inf, df, name, dic, results_dir, map_color):
    '''
    inf:            source influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream out edge aggregated weight dataframe"
    results_dir:    string of results dir
    '''
    aggr_type = str(inf + '_*')
    df = df.drop(['actors'], axis=1)
    print(df.head())
    print(df.columns)

    # Data
    heatmap = np.empty((5, len(df[aggr_type])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data

    # renaming the first label in y-axis
    df = df.rename(columns={f'{inf}_TM':f'{inf} -> TM', f'{inf}_UM':f'{inf} -> UM',f'{inf}_TF':f'{inf} -> TF',
                            f'{inf}_UF':f'{inf} -> UF', aggr_type:f'{inf} -> * *'})
    cbar_column = f'{inf}_(TM,UM,TF,UF)'

    # Plotting
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmin= 0, vmax=20, cmap=f'{map_color}')
    #im = ax.imshow(heatmap, interpolation='nearest', vmax=df[f'{inf} --> * *'].max(), cmap=f'{map_color}')
    ax.set_yticks(range(5))
    labels = df.columns
    ax.set_yticklabels(labels, rotation = 45, fontsize = 40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 35)
    ax.set_xlabel('Sorted actors according to the strength of influence', fontsize = 40)
    ax.set_title(f'{dic[name]} {inf} source actors influence', fontsize = 40)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label('Transfer Entropy', fontsize = 40)
    ax.set_aspect('auto')
    plt.savefig(f'{results_dir}/{name}_{inf}_out_activity.png')

######################################################
### creating plots for the same sources (2 Layers) ###
######################################################
def plot_user_heatmap_outdegree_2layers(inf, df, name, dic, results_dir, map_color):
    '''
    inf:            source influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream out edge aggregated weight dataframe"
    results_dir:    string of results dir
    '''
    aggr_type = str(inf + '_*')
    df = df.drop(['actors'], axis=1)
    print(df.head())
    print(df.columns)

    # Data
    heatmap = np.empty((3, len(df[aggr_type])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data

    # renaming the first label in y-axis
    df = df.rename(columns={f'{inf}_T':f'{inf} * -> T *', f'{inf}_U':f'{inf} * -> U *', aggr_type:f'{inf} * -> * *'})
    cbar_column = f'{inf}_(T,U)'

    # Plotting
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmin=0, vmax=20, cmap=f'{map_color}')
    #im = ax.imshow(heatmap, interpolation='nearest', vmax=df[f'{inf} * --> * *'].max(), cmap=f'{map_color}')
    ax.set_yticks(range(3))
    labels = df.columns
    ax.set_yticklabels(labels, rotation = 45, fontsize = 40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 35)
    ax.set_xlabel('Sorted actors according to the strength of influence', fontsize = 40)
    if inf == 'T':
        ax.set_title(f'{dic[name]} Trustworthy source actors influence', fontsize = 40)
    elif inf == 'U':
        ax.set_title(f'{dic[name]} Untrustworthy source actors influence', fontsize = 40)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label('Transfer Entropy', fontsize = 20)
    ax.set_aspect('auto')
    plt.savefig(f'{results_dir}/{name}_{inf}_out_activity.png')

######################################################
### creating plots for the Accumulated Dataframes ###
######################################################
def plot_user_heatmap_datasets_comparison(inf, df, results_dir, map_color):
    '''
    inf:            source influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream out edge aggregated weight dataframe"
    results_dir:    string of results dir
    '''
    # Data
    heatmap = np.empty((2, len(df['Skripal'])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data

    # Plotting
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmin= 0, vmax=20, cmap=f'{map_color}')
    ax.set_yticks(range(2))
    labels = df.columns
    ax.set_yticklabels(labels, rotation = 45, fontsize = 40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 35)
    ax.set_xlabel('Sorted actors according to the strength of influence', fontsize = 40)
    if inf == 'T':
        ax.set_title(f'Trustworthy source actors influence over datasets', fontsize = 40)
    elif inf == 'U':
        ax.set_title(f'Untrustworthy source actors influence over datasets', fontsize = 40)
    elif inf in ['TM', 'TF', 'UM', 'UF']:
        ax.set_title(f'{inf} source actors influence over datasets', fontsize = 40)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label('Transfer Entropy', fontsize = 30)
    ax.set_aspect('auto')
    plt.savefig(f'{results_dir}/{inf}_sources_datasets_comparison.png')

# creating plots for the same targets

def plot_layer_heatmap_4layers(df, name, dic, results_dir):
    new_indecies = ['TM_(TM,UM,TF,UF)', 'TF_(TM,UM,TF,UF)', 'UM_(TM,UM,TF,UF)', 'UF_(TM,UM,TF,UF)']
    df.set_index([new_indecies], inplace=True)
    new_columns = {'TM_*':'TM_(TM,UM,TF,UF)', 'TF_*':'TF_(TM,UM,TF,UF)', 'UM_*':'UM_(TM,UM,TF,UF)', 'UF_*':'UF_(TM,UM,TF,UF)'}
    df.rename(columns = new_columns, inplace=True)
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, vmax=1, annot=True, fmt='.2f')
    ax.set_title(f'Pairwise Link Presense between Influence Types in {dic[name]} Dataset', fontsize = 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize = 6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize = 6)
    plt.savefig(f'{results_dir}/{name}_layer_correlation_heatmap_4layers.png')

def plot_layer_heatmap_4layers_weighted(df, name, dic, results_dir):
    new_indecies = ['TM_(TM,UM,TF,UF)', 'TF_(TM,UM,TF,UF)', 'UM_(TM,UM,TF,UF)', 'UF_(TM,UM,TF,UF)']
    df.set_index([new_indecies], inplace=True)
    new_columns = {'TM_*':'TM_(TM,UM,TF,UF)', 'TF_*':'TF_(TM,UM,TF,UF)', 'UM_*':'UM_(TM,UM,TF,UF)', 'UF_*':'UF_(TM,UM,TF,UF)'}
    df.rename(columns = new_columns, inplace=True)
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, vmax=1, annot=True, fmt='.2f')
    ax.set_title(f'Pairwise Weighted Link Presence between Influence Types in {dic[name]} Dataset', fontsize = 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize = 6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize = 6)
    plt.savefig(f'{results_dir}/{name}_layer_correlation_heatmap_4layers_weighted.png')

def plot_layer_heatmap_2layers(df, name, dic, results_dir):
    new_indecies = ['T_(T,U)', 'U_(T,U)']
    df.set_index([new_indecies], inplace=True)
    new_columns = {'T_*':'T_(T,U)', 'U_*':'U_(T,U)'}
    df.rename(columns = new_columns, inplace=True)
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, vmax=1, annot=True, fmt='.2f')
    ax.set_title(f'Pairwise Link Presense between Influence Types in {dic[name]} Dataset', fontsize = 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize = 6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize = 6)
    plt.savefig(f'{results_dir}/{name}_layer_correlation_heatmap_2layers.png')

def plot_layer_heatmap_2layers_weighted(df, name, dic, results_dir):
    new_indecies = ['T_(T,U)', 'U_(T,U)']
    df.set_index([new_indecies], inplace=True)
    new_columns = {'T_*':'T_(T,U)', 'U_*':'U_(T,U)'}
    df.rename(columns = new_columns, inplace=True)
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, vmax=1, annot=True, fmt='.2f')
    ax.set_title(f'Pairwise Weighted Link Presence between Influence Types in {dic[name]} Dataset', fontsize = 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize = 6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize = 6)
    plt.savefig(f'{results_dir}/{name}_layer_correlation_heatmap_2layers_weighted.png')