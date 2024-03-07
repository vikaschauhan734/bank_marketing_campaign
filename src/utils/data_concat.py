# combine one hot encoded cols with numeric cols
import pandas as pd

def data_concat(df,cat_col):
    ohe = pd.get_dummies(df[cat_col],drop_first=True).astype(int)
    df1 = pd.concat([ohe,df.drop(cat_col,axis=1)],axis=1)
    return df1


if __name__=="__main__":
    data_concat()