# balancing the values of target value
from imblearn.over_sampling import SMOTE

def balance_data(df):
    # balancing data
    smote = SMOTE(sampling_strategy='minority')
    df1, df1['y'] = smote.fit_resample(df.drop('y',axis=1),df['y'])
    return df1