
def add_aggr_nets(df):
    '''
    From a dataframe of actors and influence-type edges, adds columns for aggregated influence 
    types (i.e. U->T from UM->TM, UM->TF, UF->TM, UF->TF)
    '''
    # Same source
    df['TM_*'] = df['TM_UF'] + df['TM_UM'] + df['TM_TF'] + df['TM_TM']
    df['TF_*'] = df['TF_UF'] + df['TF_UM'] + df['TF_TF'] + df['TF_TM']
    df['UM_*'] = df['UM_UF'] + df['UM_UM'] + df['UM_TF'] + df['UM_TM']
    df['UF_*'] = df['UF_UF'] + df['UF_UM'] + df['UF_TF'] + df['UF_TM']

    # Same source for total Trustworthy and Untrustworthy
    df['T_*'] = df['TF_*'] + df['TM_*']
    df['U_*'] = df['UF_*'] + df['UM_*']
    
    return df
