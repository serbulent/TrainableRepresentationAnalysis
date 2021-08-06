import pandas as pd
from pathlib import Path
import numpy as np
from pandas import ExcelWriter

representation_name_list = []
mcc_list = []

for path in Path("../results/").glob("dt_prot_famliy_pred_*_report_nc.csv"):
    mcc = pd.read_csv(path)['MCC'][0]
    representation_name_list.append(str(path).split('pred_')[1].split('_report_nc.csv')[0])
    #print(representation_name_list)
    mcc_list.append(mcc)
    #print(mcc_list)


df_nc = pd.DataFrame(mcc_list)
#print(representation_name_list)
df_nc = df_nc.transpose()
df_nc.columns = representation_name_list
#df_nc = df_nc.reindex(df_nc.mean().sort_values().index, axis=1)
representation_name_list = []
mcc_list = []

for path in Path("../results/").glob("dt_prot_famliy_pred_*_report_uc15.csv"):
    mcc = pd.read_csv(path)['MCC'][0]
    representation_name_list.append(str(path).split('pred_')[1].split('_report_uc15.csv')[0])
    #print(representation_name_list)
    mcc_list.append(mcc)
    #print(path)


df_uc15 = pd.DataFrame(mcc_list)

df_uc15 = df_uc15.transpose()
df_uc15.columns = representation_name_list
df_uc15 = df_uc15.reindex(df_nc.columns, axis=1)
mcc_list = []
representation_name_list = []
for path in Path("../results/").glob("dt_prot_famliy_pred_*_report_uc30.csv"):
    mcc = pd.read_csv(path)['MCC'][0]
    representation_name_list.append(str(path).split('pred_')[1].split('_report_uc30.csv')[0])
    #print(representation_name_list)
    mcc_list.append(mcc)
    #print(path)


df_uc30 = pd.DataFrame(mcc_list)

df_uc30 = df_uc30.transpose()
df_uc30.columns = representation_name_list
df_uc30 = df_uc30.reindex(df_nc.columns, axis=1)
mcc_list = []
representation_name_list = []
for path in Path("../results/").glob("dt_prot_famliy_pred_*_report_uc50.csv"):
    mcc = pd.read_csv(path)['MCC'][0]
    representation_name_list.append(str(path).split('pred_')[1].split('_report_uc50.csv')[0])
    #print(representation_name_list)
    mcc_list.append(mcc)
    #print(path)


df_uc50 = pd.DataFrame(mcc_list)

df_uc50 = df_uc50.transpose()
df_uc50.columns = representation_name_list
df_uc50 = df_uc50.reindex(df_nc.columns, axis=1)
#print(representation_name_list)

df_all = pd.concat([df_nc, df_uc15, df_uc30, df_uc50])
df_all = df_all.reindex(['BLAST','HMMER','K-SEP','APAAC','PFAM^','AAC','PROTVEC',\
                            'GENE2VEC^','LEARNED-VEC','MUT2VEC^','TCGA-EMBEDDING','SEQVEC','CPC-PROT','BERT-BFD',\
                            'BERT-PFAM^','ESMB1','ALBERT','XLNET','UNIREP','T5'], axis=1)

df_all.insert(loc=0, column='', value=['No cluster', 'Uniclust15', 'Uniclust30', 'Uniclust50'])
df_all = df_all.transpose()
writer = ExcelWriter('../results/mcc_table.xlsx')
df_all.to_excel(writer,'Sheet1')
writer.save()
df_all.to_csv('../results/mcc_table.csv')
print(df_all)
