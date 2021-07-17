import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from pathlib import Path

ac_list = []
f1_list = []
mcc_list = []
representation_name_list = []
for path in Path("../results/").glob("drug_target_family_pred_f1_[!p]*_uc50.npy"):
    #print(path)
    representation_name_list.append(str(path).split("f1_")[1].split("_uc50.npy")[0])
    f1s = np.load(path)
    df = pd.DataFrame(f1s)
    f1_list.append(df)

df_f1 = pd.concat(f1_list, axis=1)
df_f1.columns = representation_name_list
#df_f1 = df_f1.reindex(df_f1.mean().sort_values().index, axis=1)
representation_name_list = []
for path in Path("../results/").glob("drug_target_family_pred_accuracy_[!p]*_uc50.npy"):
    #print(path)i
    representation_name_list.append(str(path).split("accuracy_")[1].split("_uc50.npy")[0])
    acs = np.load(path)
    df = pd.DataFrame(acs)
    ac_list.append(df)

df_ac = pd.concat(ac_list, axis=1)
df_ac.columns = representation_name_list
#df_ac = df_ac.reindex(df_ac.mean().sort_values().index, axis=1)
representation_name_list = []
for path in Path("../results/").glob("drug_target_family_pred_mcc_[!p]*_uc50.npy"):
    #print(path)
    representation_name_list.append(str(path).split("mcc_")[1].split("_uc50.npy")[0])
    mccs = np.load(path)
    df = pd.DataFrame(mccs)
    mcc_list.append(df)

df_mcc = pd.concat(mcc_list, axis=1)
df_mcc.columns = representation_name_list
df_mcc = df_mcc.reindex(df_mcc.mean().sort_values().index, axis=1)
df_f1 = df_f1.reindex(columns=df_mcc.columns)
df_ac = df_ac.reindex(columns=df_mcc.columns)

df_f1['metric'] = 'f1-score'
df_ac['metric'] = 'accuracy'
df_mcc['metric'] = 'mcc'
#print(df_mcc)

all_data = pd.concat([df_ac, df_f1, df_mcc])
#all_data = all_data.reindex(columns=df_mcc.columns)
#print(all_data)
cols = all_data.columns
cols = cols[:-1]
#print(cols)
all_data = pd.melt(all_data, id_vars=['metric'], value_vars=cols)



group_color_dict = {'K-SEP':'green','BERT-PFAM^':'red', 'UNIREP':'red', 'T5':'red', 'BERT-BFD':'red',\
                     'SEQVEC':'red', 'ALBERT':'red', 'PFAM^':'green', 'ESMB1':'red', \
                     'XLNET':'red', 'AAC':'green', 'APAAC':'green', 'PROTVEC':'blue', 'MUT2VEC':'blue',\
                    'LEARNED-VEC':'blue', 'CPC-PROT':'blue', 'BLAST':'blue', 'TCGA-EMBEDDING':'blue', 'GENE2VEC^':'blue', 'MUT2VEC^':'blue', 'HMMER':'blue'}


def set_colors_and_marks_for_representation_groups(ax):
    for label in ax.get_yticklabels():
        label.set_color(group_color_dict[label.get_text()])
        if label.get_text() == 'PFAM' or label.get_text() == 'BERT-PFAM' :
            signed_text = label.get_text() + "*"
            label.set_text(signed_text)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

sns.set(rc={'figure.figsize':(13.7,18.27)})
sns.set_theme(style="whitegrid", color_codes=True)

ax = sns.boxplot(data=all_data, x=all_data['value'], y=all_data['variable'], hue=all_data['metric'], whis=np.inf,  orient="h")
#ax = sns.swarmplot(data=all_data, x=all_data['value'], y=all_data['variable'], orient="h",color=".1")
#ax = sns.boxplot(data=df_ac, whis=np.inf,  orient="h")
#ax = sns.swarmplot(data=df_ac, orient="h",color=".1")

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', color='gainsboro', linewidth=1.0)
ax.grid(b=True, which='minor', color='whitesmoke', linewidth=0.5)

yticks = ax.get_yticks()
for ytick in yticks:
    ax.hlines(ytick+0.5,-0.1,1,linestyles='dashed')


set_colors_and_marks_for_representation_groups(ax)
ax.get_figure().savefig('../results/figures/general_uc50.png')

