
def maximum_occurring(df,col):
    # replace unknown value with most occuring value with the column
    return df[col].replace("unknown",df[col].value_counts().idxmax()).infer_objects(copy=False)

def clean_col(df,col):
    df[col] = maximum_occurring(df, col)
    return df[col].replace({'yes':1, 'no':0}).infer_objects(copy=False)

if __name__=="__main__":
    clean_col()


