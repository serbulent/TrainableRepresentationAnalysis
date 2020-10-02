import pandas as pd
from os import listdir

#--calculating size, single-label count and multi-label count of each dataset--

files = listdir(r"datasets\final")

df_stat = pd.DataFrame(columns=["dataset_name","size","single-label_count","multi-label_count"])
for f in files:
    df = pd.read_csv(r"datasets\final\{}".format(f),sep="\t")
    len_df = len(df)
    df_single = df.loc[df.sum(axis=1)==1]
    df_multi = df.loc[df.sum(axis=1)>1]
    df_multi = df_multi.iloc[:,1:]
    vectors = df_multi.values
 
    vector_dict = {}
    for v in vectors:
        if str(v) not in vector_dict.keys():
            vector_dict[str(v)] = 1
        else: 
            vector_dict[str(v)] += 1

    print(len(df_multi),len(df_single),len_df)
    if len(df_multi)+len(df_single) != len_df:
        print("oops! there is an error -_-")
    df_stat.loc[len(df_stat)] = [f.split(".")[0],len_df,len(df_single),vector_dict]
df_stat.to_csv(r"dataset_statistics_final.tsv",sep="\t",index=None)
        
