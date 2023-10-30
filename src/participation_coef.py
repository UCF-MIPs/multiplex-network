#TODO remove base function and replace with in/out functions
import numpy as np

def part_coef(df, sour_infl):
    # probably should include actors and go row by row
    m=4
    types = ['UF', 'UM', 'TF', 'TM']
    aggr_type = str(sour_infl + '_*')
    print(aggr_type)
    o = df[aggr_type].to_numpy()
    ptemp = np.zeros_like(o)
    for i in types:
        infl_type = str(sour_infl + '_' + i)
        print(infl_type)
        k = df[infl_type].to_numpy()
        #print(np.count_nonzero(o))
        #ptemp += np.divide(k,o,out=np.zeros_like(k), where=o!=0, dtype=float)**2
        ptemp += np.divide(k,o)**2
        #print(ptemp)
    ptemp[np.isinf(ptemp)] = np.nan
    p = (4./3.)*(1-ptemp)
    return p # array of part. coeffs


### new participant coefficient functions ###
## Sources/outdegree ##
#TODO make so this doesn't overwrite previous df
def part_coef_out(df, inf, layers):
    '''
    input: df of aggregated weight of out edges
    adds column to df, alters df in place
    '''
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
    '''
    input: df of aggregated weight of in edges
    adds column to df, alters df in place
    '''
    types = ['UM','TM','UF','TF']
    aggr_type = str('*_' + inf)
    df['ptemp'] = 0
    for i in types:
        infl_type = str(i + '_' + inf)
        df['ptemp'] += np.power(np.divide(df[infl_type],df[aggr_type]),2)
    df['pc_in'] = (layers/(layers-1))*(1-df['ptemp'])
    return df

