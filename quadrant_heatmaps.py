import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from src import plot_heatmap

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']
for dataset in dataset_list:
    df_source_path = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources.csv'
    df_heatmap_source = pd.read_csv(df_source_path, index_col=0)
    plot_heatmap.plot_layer_heatmap(df_heatmap_source, dataset, f'plots/{dataset}/Sources')
    plt.clf()

for dataset in dataset_list:
    df_target_path = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_targets.csv'
    df_heatmap_target = pd.read_csv(df_target_path, index_col=0)
    plot_heatmap.plot_layer_heatmap(df_heatmap_target, dataset, f'plots/{dataset}/Targets')
    plt.clf()