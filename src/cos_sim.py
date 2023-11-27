import numpy as np
from numpy.linalg import norm


def cos_sim_dfs(df1, df2):
    '''
    takes in two dataframes, 
    makes sure the columns are appropriate, 
    then returns cos similarity value

    '''
    #TODO check that both inputs are dfs, check labels if necessary 
    a = df1.to_numpy().flatten()
    b = df2.to_numpy().flatten()
    c_s = np.dot(a,b)/(norm(a)*norm(b))
    return c_s


def cos_sim(a,b):
    '''
    takes in two numpy arrays
    ...
    '''
    a = a.flatten()
    b = b.flatten()
    c_s = np.dot(a,b)/(norm(a)*norm(b))
    return c_s

