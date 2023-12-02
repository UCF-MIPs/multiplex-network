import numpy as np

### new participant coefficient functions ###
## Sources/outdegree ##
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


