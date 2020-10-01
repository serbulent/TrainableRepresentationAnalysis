import pandas as pd
import os


#---multilabel dataset construction---

df_category = pd.read_csv(r"go_category_dataframe_dis-terms.tsv",sep="\t",converters={"Dissimilar_Terms":lambda x:x.strip("{}").replace("'","").split(", ")})

#-excluding go categories not having dissimilar go terms-
df_cat_notnull = df_category.loc[df_category["Dissimilar_Terms"].apply(len)!=1].reset_index()  


#-generating datasets by merging protein accessions to corresponding go terms-

df_goterm = pd.read_csv(r"dissimilar_terms_dataframe.tsv",sep="\t",converters={"Proteins":lambda x:x.strip("[]").replace("'","").split(", ")})

for i in range(len(df_cat_notnull)):
    df =  df_goterm.loc[df_goterm["GO_ID"].isin(df_cat_notnull["Dissimilar_Terms"][i])].reset_index()

    prot = set([pt for pt_list in df["Proteins"] for pt in pt_list])
    train = pd.DataFrame(columns=["Protein_Id"]+[go for go in df["GO_ID"]])
    for pt1 in prot:
        pt_label = []
        for pt2 in df["Proteins"]:
            if pt1 in pt2:
                pt_label.append(1)
            else:
                pt_label.append(0)
        train.loc[len(train)] = [pt1]+pt_label
    if df_cat_notnull["Aspect"][i] == "cellular_component":
        aspect_label = "CC"
    elif df_cat_notnull["Aspect"][i] == "biological_process":
        aspect_label = "BP"        
    elif df_cat_notnull["Aspect"][i] == "molecular_function":
        aspect_label = "MF"

    train.to_csv(r"datasets\initial\{0}_{1}_{2}.tsv".format(aspect_label,df_cat_notnull["Number_Category"][i],df_cat_notnull["Term_Specificity"][i]),sep="\t",index=None)
                 

#--filtering datasets by keeping only proteins which have representations for all descriptors--

features = os.listdir(r"protein_representations\final")

ft_init = pd.read_csv(r"protein_representations\final\{0}".format(features[0]),sep="\t",usecols=["Protein_Id"])

common_prots = set(ft_init["Protein_Id"])
for ft in features[1:]:
    df_ft = pd.read_csv(r"protein_representations\final\{0}".format(ft),sep="\t",usecols=["Protein_Id"])
    common_prots = common_prots.intersection(set(df_ft["Protein_Id"]))
    
datasets = os.listdir(r"datasets\initial") 
for dt in datasets:
    df_dt = pd.read_csv(r"datasets\initial\{0}".format(dt),sep="\t")
    df_dt_final = df_dt.loc[df_dt["Protein_Id"].isin(common_prots)]
    print("initial-final: {0} - {1}".format(len(df_dt),len(df_dt_final)))
    print("lost_percentage: {0}".format(((len(df_dt)-len(df_dt_final))/len(df_dt))*100))
    df_dt_final.to_csv(r"datasets\final\{0}".format(dt),sep="\t",index=None)
    






