#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:17:41 2024

@author: hiro
"""


def clean():
    import os
    import pandas as pd
    import numpy as np

    
    os.chdir('/Users/hiro/DS-ML')
    df = pd.read_csv('data_30countries.csv')

#dd = pd.read_csv('c13k_selections.csv')
    """
    indexes_to_drop = []
    
    for i in df['subject_global'].unique():
        df_temporary = df[df['subject_global'] == i]
        if len(df_temporary) != 28:
            indexes_to_drop.append(i)
            
    df = df[~df['subject_global'].isin(indexes_to_drop)]

    original_array = df['subject_global']
    
    # Get the unique values and create a mapping to consecutive integers
    unique_values = np.unique(original_array)
    mapping = {old: new for new, old in enumerate(unique_values)}
    
    filled_array = np.array([mapping[val] for val in original_array])
    
    df['subject_global'] = filled_array
    """
    dataframes = {}
    
    for i in range(0,17):
        if i <= 15:
            dataframes[f'df_{i}'] = df.iloc[:, i*10:i*10+9]
        if i == 16:
            dataframes[f'df_{i}'] = df.iloc[:, 160:165]
    
    df0 = dataframes['df_0']
    
    df1 = dataframes['df_1']
    df2 = dataframes['df_2']
    df3 = dataframes['df_3']
    df4 = dataframes['df_4']
    df5 = dataframes['df_5']
    df6 = dataframes['df_6']
    df7 = dataframes['df_7']
    df8 = dataframes['df_8']
    df9 = dataframes['df_9']
    df10 = dataframes['df_10']
    df11 = dataframes['df_11']
    df12 = dataframes['df_12']
    df13 = dataframes['df_13']
    df14 = dataframes['df_14']
    df15 = dataframes['df_15']
    df16 = dataframes['df_16']

    # Original array
    
    

    
    #testing = df0[df0['equiv_nr'] == 3].copy()
    #training = df0[df0['equiv_nr'] != 3].copy()

    d = df0[["subject_global", "high", "low", "probability", "equivalent", "equiv_nr"]]

    d.loc[d['equiv_nr'] == 44, 'low'] = d['equivalent'].astype(d['low'].dtype)
    d.loc[d['equiv_nr'] == 44, 'equivalent'] = 0

    d.columns = ['subject_global', 'x', 'y', 'p', 'ce', 'equiv_nr']
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(d, test_size=0.2, random_state=42)

    

    return d, train_df, test_df


def clean2():
    
    import os
    
    
    from sklearn.model_selection import train_test_split
    
    import numpy as np
    import pandas as pd
    
    os.chdir('/Users/hiro/DS-ML')
    from DS_ML_Data import clean
    
    d, train_df, test_df = clean()
    
    ce_list = []
    
    #aa = d.columns.tolist()
    
    #bb = [f"ce_{i}" for i in range(28)]
    
    #aa = aa + bb
    
    
    #for i in bb:
    #    d[i] = np.nan
    
    
    indexes_to_drop = []
    
    for i in d['subject_global'].unique():
        df_temporary = d[d['subject_global'] == i]
        if len(df_temporary) != 28:
            indexes_to_drop.append(i)
            
    d = d[~d['subject_global'].isin(indexes_to_drop)]
    
    updated_rows = []
    
    for i in d['subject_global'].unique():
        df_temporary = d[d['subject_global'] == i].copy()
        df_temporary = df_temporary[['subject_global', 'x', 'y', 'p', 'ce', 'equiv_nr']]
    
        # Create the repeated CE DataFrame
        z = np.array(df_temporary[['ce']].T)[0]
        u = pd.DataFrame(z).T
        k = pd.DataFrame(np.repeat(u, len(df_temporary), axis=0))
        np.fill_diagonal(k.values, np.nan)
        k.columns = [f"ce_{j}" for j in range(28)]
    
        # Concatenate the original df_temporary and the new CE columns
        merged_df = pd.concat([df_temporary.reset_index(drop=True), k.reset_index(drop=True)], axis=1)
    
        # Collect updated rows to reassemble the full DataFrame later
        updated_rows.append(merged_df)
    
    # Step 3: Reconstruct the full DataFrame with updated values
    d_updated = pd.concat(updated_rows, ignore_index=True)

    columns = [col for col in d_updated.columns if col != 'ce'] + ['ce']
    d_updated = d_updated[columns]

    train_df_updated, test_df_updated = train_test_split(d_updated, test_size=0.2, random_state=42)

    return d_updated, train_df_updated, test_df_updated
