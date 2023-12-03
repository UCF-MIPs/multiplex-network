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
dict_list = {'Skripal': 'Skripal', 'Ukraine': 'Ukraine', 'Anniversary': 'Anniversary', 'Biden': 'Biden', 'Bucha_crimes': 'Bucha Crimes',
             'crimes_un_report': 'Crimes UN Report', 'Khersion_retreat': 'Khersion Retreat', 'Mariupol_hospital': 'Mariupol Hospital',
             'Mariupol_theater': 'Mariupol Theater', 'Putin_warrant': 'Putin Warrant', 'Russia_mobilize': 'Russia Mobilize', 
             'Russian_missle_cross_Poland': 'Russian Missle Cross Poland', 'tanks': 'Tanks', 
             'Zelensky_visit_the_US': 'Zelensky Visit the US'}
dataframes = {}
for dataset in dataset_list: 
    df_name = f'df_{dataset}'
    df_path = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_4layers_weighted.csv'
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
    combined_dfs[df_name].to_csv(f'Data/preprocessed/{inf}_cosine_similarity_raw_weighted.csv')

# cosine similarity for TM
TM_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
TM_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        TM_array = combined_dfs['combined_df_TM'].values
        TM_cosine_similarity_df.at[j,h] = consine_sim(TM_array[i,:], TM_array[k,:]).round(3)
TM_cosine_similarity_df.rename(columns=dict_list, inplace=True)
TM_cosine_similarity_df.index = dict_list.values()
TM_cosine_similarity_df.to_csv('results/TM_cosine_similarity_weighted.csv')


# cosine similarity for TF
TF_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
TF_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        TF_array = combined_dfs['combined_df_TF'].values
        TF_cosine_similarity_df.at[j,h] = consine_sim(TF_array[i,:], TF_array[k,:]).round(3)
TF_cosine_similarity_df.rename(columns=dict_list, inplace=True)
TF_cosine_similarity_df.index = dict_list.values()
TF_cosine_similarity_df.to_csv('results/TF_cosine_similarity_weighted.csv')


# cosine similarity for UM
UM_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
UM_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        UM_array = combined_dfs['combined_df_UM'].values
        UM_cosine_similarity_df.at[j,h] = consine_sim(UM_array[i,:], UM_array[k,:]).round(3)
UM_cosine_similarity_df.rename(columns=dict_list, inplace=True)
UM_cosine_similarity_df.index = dict_list.values()
UM_cosine_similarity_df.to_csv('results/UM_cosine_similarity_weighted.csv')

# cosine similarity for UF
UF_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
UF_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        UF_array = combined_dfs['combined_df_UF'].values
        UF_cosine_similarity_df.at[j,h] = consine_sim(UF_array[i,:], UF_array[k,:]).round(3)
UF_cosine_similarity_df.rename(columns=dict_list, inplace=True)
UF_cosine_similarity_df.index = dict_list.values()
UF_cosine_similarity_df.to_csv('results/UF_cosine_similarity_weighted.csv')


### Cosine similarity for Trustworthy and Untrustworthy ###
dataframes_TU = {}
for dataset in dataset_list: 
    df_name = f'df_{dataset}'
    df_path = f'Data/preprocessed/{dataset}/{dataset}_quadrant_preprocessing_sources_2layers_weighted.csv'
    dataframes_TU[df_name] = pd.read_csv(df_path, index_col=0)

combined_dfs_TU = {}
for inf in ['T','U']:
    df_name = f'combined_df_{inf}'
    combined_dfs_TU[df_name] = pd.DataFrame(columns = ['T_*','U_*'])
    if inf == 'T':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs_TU[df_name].loc[i] = dataframes_TU[f'df_{j}'].iloc[0]
    elif inf == 'U':
        for i,j in zip(range(len(dataset_list)), dataset_list):
            combined_dfs_TU[df_name].loc[i] = dataframes_TU[f'df_{j}'].iloc[1]
    combined_dfs_TU[df_name].to_csv(f'Data/preprocessed/{inf}_cosine_similarity_raw_weighted.csv')

# cosine similarity for Trustworthy
T_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
T_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        T_array = combined_dfs_TU['combined_df_T'].values
        T_cosine_similarity_df.at[j,h] = consine_sim(T_array[i,:], T_array[k,:]).round(3)
T_cosine_similarity_df.rename(columns=dict_list, inplace=True)
T_cosine_similarity_df.index = dict_list.values()
T_cosine_similarity_df.to_csv('results/T_cosine_similarity_weighted.csv')

# cosine similarity for Untrustworthy
U_cosine_similarity_df = pd.DataFrame(index= range(len(dataset_list)), columns = dataset_list)
U_cosine_similarity_df.index = dataset_list
for i,j in enumerate(dataset_list):
    for k,h in enumerate(dataset_list):
        U_array = combined_dfs_TU['combined_df_U'].values
        U_cosine_similarity_df.at[j,h] = consine_sim(U_array[i,:], U_array[k,:]).round(3)
U_cosine_similarity_df.rename(columns=dict_list, inplace=True)
U_cosine_similarity_df.index = dict_list.values()
U_cosine_similarity_df.to_csv('results/U_cosine_similarity_weighted.csv')