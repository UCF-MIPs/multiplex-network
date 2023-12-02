import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def consine_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

dataset_list = ['Skripal', 'Ukraine', 'Anniversary', 'Biden', 'Bucha_crimes','crimes_un_report','Khersion_retreat',
                'Mariupol_hospital','Mariupol_theater','Putin_warrant','Russia_mobilize','Russian_missle_cross_Poland',
                'tanks','Zelensky_visit_the_US']
dataframes = {}
for dataset in dataset_list: 
    df_name = f'df_{dataset}'
    df_path = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources.csv'
    dataframes[df_name] = pd.read_csv(df_path, index_col=0)

combined_dfs = {}
for inf in ['TM','TF','UM','UF']:
    df_name = f'combined_df_{inf}'
    combined_dfs[df_name] = pd.DataFrame(columns = ['TM_*','TF_*','UM_*','UF_*'])
    if inf == 'TM':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs[df_name].loc[i] = dataframes[f'df_{j}'].iloc[0]
    elif inf == 'TF':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs[df_name].loc[i] = dataframes[f'df_{j}'].iloc[1]
    elif inf == 'UM':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs[df_name].loc[i] = dataframes[f'df_{j}'].iloc[2]
    elif inf == 'UF':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs[df_name].loc[i] = dataframes[f'df_{j}'].iloc[3]
    combined_dfs[df_name].to_csv(f'Data/preprocessed/{inf}_cosine_similarity_raw.csv')

# cosine similarity for TM
TM_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
TM_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        TM_array = combined_dfs['combined_df_TM'].values
        TM_cosine_similarity_df.at[j,h] = consine_sim(TM_array[i,:], TM_array[k,:]).round(3)
TM_cosine_similarity_df.to_csv('Data/preprocessed/TM_cosine_similarity.csv')
# TM_table = plt.table(cellText=TM_cosine_similarity_df.values, rowLabels=TM_cosine_similarity_df.index,\
#                       colLabels=TM_cosine_similarity_df.columns, cellloc='center')
# plt.savefig(f'plots/TM_cosine_similarity_table.png')
# plt.clf()

# cosine similarity for TF
TF_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
TF_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        TF_array = combined_dfs['combined_df_TF'].values
        TF_cosine_similarity_df.at[j,h] = consine_sim(TF_array[i,:], TF_array[k,:]).round(3)
TF_cosine_similarity_df.to_csv('Data/preprocessed/TF_cosine_similarity.csv')
# TF_table = plt.table(cellText=TF_cosine_similarity_df.values, rowLabels=TF_cosine_similarity_df.index,\
#                       colLabels=TF_cosine_similarity_df.columns, loc='center')
# plt.savefig(f'plots/TF_cosine_similarity_table.png')
# plt.clf()

# cosine similarity for UM
UM_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
UM_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        UM_array = combined_dfs['combined_df_UM'].values
        UM_cosine_similarity_df.at[j,h] = consine_sim(UM_array[i,:], UM_array[k,:]).round(3)
UM_cosine_similarity_df.to_csv('Data/preprocessed/UM_cosine_similarity.csv')
# UM_table = plt.table(cellText=UM_cosine_similarity_df.values, rowLabels=UM_cosine_similarity_df.index,\
#                       colLabels=UM_cosine_similarity_df.columns, loc='center')
# plt.savefig(f'plots/UM_cosine_similarity_table.png')
# plt.clf()

# cosine similarity for UF
UF_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
UF_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        UF_array = combined_dfs['combined_df_UF'].values
        UF_cosine_similarity_df.at[j,h] = consine_sim(UF_array[i,:], UF_array[k,:]).round(3)
UF_cosine_similarity_df.to_csv('Data/preprocessed/UF_cosine_similarity.csv')
# UF_table = plt.table(cellText=UF_cosine_similarity_df.values, rowLabels=UF_cosine_similarity_df.index,\
#                       colLabels=UF_cosine_similarity_df.columns, loc='center')
# plt.savefig(f'plots/UF_cosine_similarity_table.png')
# plt.clf()
