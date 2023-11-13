import numpy as np
from numpy.linalg import norm


def cos_sim(df1, df2):
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

