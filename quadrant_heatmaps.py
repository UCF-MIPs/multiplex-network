import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src import plot_heatmap

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US', 'Navalny']
dict_list = {'Skripal': 'Skripal', 'Ukraine': 'Ukraine', 'Anniversary': 'Anniversary', 'Biden': 'Biden', 'Bucha_crimes': 'Bucha Crimes',
             'crimes_un_report': 'Crimes UN Report', 'Khersion_retreat': 'Khersion Retreat', 'Mariupol_hospital': 'Mariupol Hospital',
             'Mariupol_theater': 'Mariupol Theater', 'Putin_warrant': 'Putin Warrant', 'Russia_mobilize': 'Russia Mobilize', 
             'Russian_missle_cross_Poland': 'Russian Missle Cross Poland', 'tanks': 'Tanks', 
             'Zelensky_visit_the_US': 'Zelensky Visit the US', 'Navalny': 'Navalny'}
for dataset in dataset_list:
    df_path_4 = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_4layers.csv'
    df_heatmap_source = pd.read_csv(df_path_4, index_col=0)
    plot_heatmap.plot_layer_heatmap_4layers(df_heatmap_source, dataset, dict_list, f'plots/{dataset}/TUMF')
    plt.clf()

    df_path_4_w = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_4layers_weighted.csv'
    df_heatmap_source = pd.read_csv(df_path_4_w, index_col=0)
    plot_heatmap.plot_layer_heatmap_4layers_weighted(df_heatmap_source, dataset, dict_list, f'plots/{dataset}/TUMF')
    plt.clf()

    df_path_2 = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_2layers.csv'
    df_heatmap_source = pd.read_csv(df_path_2, index_col=0)
    plot_heatmap.plot_layer_heatmap_2layers(df_heatmap_source, dataset, dict_list, f'plots/{dataset}/TU')
    plt.clf()

    df_path_2_w = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_2layers_weighted.csv'
    df_heatmap_source = pd.read_csv(df_path_2_w, index_col=0)
    plot_heatmap.plot_layer_heatmap_2layers_weighted(df_heatmap_source, dataset, dict_list, f'plots/{dataset}/TU')
    plt.clf()