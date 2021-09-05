# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from Bio import SeqIO


#---UniRep convert to tsv---#
array = np.load(r"UniRep\Unirep_calculated_human_protein_vectors.npy",allow_pickle=True)
array_dict = array.item()

df = pd.DataFrame(data=[[i]+array_dict[i].tolist() for i in array_dict.keys()])
df.to_csv(r"UniRep\Unirep_calculated_human_protein_vectors.tsv",sep="\t",index=None)

#---Learned_Embed formatting---#
df1 = pd.read_csv(r"raw\learned_embed_dim64.csv",header=None,names=["target_id","features"],converters={1:lambda x:x.strip('"[]"').split(", ")})
df2 = pd.DataFrame(df1.iloc[:,1].values.tolist())
df3 = pd.concat([df1.iloc[:,0],df2],axis=1)
df3.to_csv(r"raw\learned_embed_dim64.csv",index=None)


##---Bepler, Lstm, Resnet, Transformer formatting---#
def tape_formatting(prot_rep):
    df1 = pd.read_csv(r"raw\embedding_dataframes\{}_dataframe.csv".format(prot_rep),converters={"Vector":lambda x:x.strip('[]').split()},usecols=["Entry","Vector"])
    df2 = pd.DataFrame(df1.iloc[:,1].values.tolist())
    df3 = pd.concat([df1.iloc[:,0],df2],axis=1)
    print(len(df1),len(df2),len(df3))
    dim = len(df3.columns)-1

    df3.columns = ["Protein_Id"]+["D{}".format(str(i)) for i in range(1,len(df3.columns))]

    uniprot = pd.read_csv(r"uniprot_human_all.tab",sep="\t",usecols=["Entry","Entry name"])
    uniprot.rename(columns={"Entry":"Protein_Id", "Entry name":"Protein_Name"},inplace=True)

    df4 = uniprot.merge(df3,on="Protein_Id")
    df4.to_csv(r"final\{0}_dim{1}.tsv".format(prot_rep,dim),sep="\t",index=None)

bepler = tape_formatting("bepler")
lstm = tape_formatting("lstm")
resnet = tape_formatting("resnet")
transformer = tape_formatting("transformer")
    
#-resnet rescaling feature values-    
resnet = pd.read_csv(r"final\resnet_dim256.tsv",sep="\t")
resnet.iloc[:,2:] = resnet.iloc[:,2:].values*(10**-5)
resnet.to_csv(r"final\resnet-rescaled_dim256.tsv",sep="\t",index=None)



#---id mapping---#

def uniprot_nameToid(path,sep,header):
    uniprot = pd.read_csv(r"uniprot_human_all.tab",sep="\t",usecols=["Entry","Entry name"])
    uniprot.rename(columns={"Entry":"Protein_Id", "Entry name":"Protein_Name"},inplace=True)

    feature = pd.read_csv(r"raw\{}".format(path),sep=sep, header=header)
    feature.columns = ["Protein_Name"]+["D{}".format(str(i)) for i in range(1,len(feature.columns))]

    df = uniprot.merge(feature, on=["Protein_Name"])
    df.to_csv(r"final\{}.tsv".format(path.split(".")[0]),sep="\t",index=None)
    
learned_embed = uniprot_nameToid('learned_embed_dim64.csv', ",", header=0)    
protvec = uniprot_nameToid('protvec_dim100.csv', "\t", header=None)   
seqvec = uniprot_nameToid('seqvec_dim1024.csv', ",", header=None)   
unirep = uniprot_nameToid('unirep_dim5700.tsv', "\t", header=0)   

#--
    
def uniprot_geneToid(path,sep,header):  
    uniprot = pd.read_csv(r"uniprot_human_all.tab",sep="\t",usecols=["Entry","Gene names"],converters={"Gene names":lambda x:x.split(" ")})
    uniprot.rename(columns={"Entry":"Protein_Id"},inplace=True)

    feature = pd.read_csv(r"raw\{}".format(path),sep=sep, header=header)
    feature.columns = ["Gene_Name"]+["D{}".format(str(i)) for i in range(1,len(feature.columns))]

    uniprot["Gene_Name"] = [None]*len(uniprot)
    for i in range(len(uniprot)):
        for j in uniprot["Gene names"][i]:
            if j in list(feature["Gene_Name"]):
                uniprot["Gene_Name"][i] = j
                break

    uniprot = uniprot[["Protein_Id","Gene_Name"]]
    uniprot = uniprot.loc[uniprot["Gene_Name"].notnull()]

    df = uniprot.merge(feature, on=["Gene_Name"])
    df.to_csv(r"final\{}.tsv".format(path.split(".")[0]),sep="\t",index=None)    

gene2vec = uniprot_geneToid('gene2vec_dim200.txt',sep="\s+",header=None)
tcga = uniprot_geneToid('tcga_dim50.csv',sep=",",header=0)


def ensemblTouniprot(path,sep,header):
    ensembl = pd.read_csv(r"ensembl2uniprot.tab",sep="\t",usecols=["Protein_Id","Ensembl_Id"])
    ensembl = ensembl[["Protein_Id","Ensembl_Id"]]

    feature = pd.read_csv(r"raw\{}".format(path),sep=sep, header=header,usecols=[i for i in range(0,301)])
    feature.columns = ["Ensembl_Id"]+["D{}".format(str(i)) for i in range(1,len(feature.columns))]

    df = ensembl.merge(feature, on=["Ensembl_Id"])
    df.to_csv(r"final\{}.tsv".format(path.split(".")[0]),sep="\t",index=None)    

mut2vec = ensemblTouniprot('mut2vec_dim300.txt',sep=" ",header=None)
    


#---apaac formatting---
df = pd.read_csv(r"raw\apaac\apaac.tsv",sep="\t",converters={"#":lambda x: x.split("|")})
prot_id = [i[1] for i in df["#"]]
prot_name = [i[2] for i in df["#"]]
df.insert(loc=1,column="Protein_Id",value=prot_id)
df.insert(loc=2,column="Protein_Name",value=prot_name)
df = df.iloc[:,1:]
df.to_csv(r"final\apaac_dim80.tsv",sep="\t",index=None)


#---ksep formatting---
df_final = pd.DataFrame()
for i in range(1,42):
    prot = pd.DataFrame(columns=["Protein_Id","Protein_Name"])
    for record in SeqIO.parse(r"raw\possum\fasta_splits\group_{0}.fasta".format(i),"fasta"):
        prot.loc[len(prot)] = [record.id.split("|")[1],record.id.split("|")[2]]
    ksep = pd.read_csv(r"raw\possum\k-sep_files\group{0}_ksep.csv".format(i))
    df = pd.concat([prot,ksep],axis=1)
    df_final = pd.concat([df_final,df],axis=0)

df_final.to_csv(r"final\ksep_dim400.tsv",sep="\t",index=None)


#---Transformer (updated) formatting---
df1 = pd.read_pickle(r"raw\transformer_dataframe_pool.pkl") #also done for transformer-avg by replacing file name
df2 = pd.DataFrame(df1.iloc[:,1].values.tolist())
df3 = pd.concat([df1.iloc[:,0],df2],axis=1)
print(len(df1),len(df2),len(df3))
dim = len(df3.columns)-1
df3.columns = ["Protein_Id"]+["D{}".format(str(i)) for i in range(1,len(df3.columns))]

uniprot = pd.read_csv(r"uniprot_human_all.tab",sep="\t",usecols=["Entry","Entry name"])
uniprot.rename(columns={"Entry":"Protein_Id", "Entry name":"Protein_Name"},inplace=True)

df4 = uniprot.merge(df3,on="Protein_Id")
df4.to_csv(r"final\transformer-pool_dim{}.tsv".format(dim),sep="\t",index=None)




#--all proteins in datasets and missing ones in updated transformer file--
df = pd.read_csv(r"final\transformer_dim768.tsv",sep="\t")   #this file is transformer-avg 
datasets = os.listdir(r"C:\Users\HEVAL\Desktop\GO_Prediction\datasets\final")

prot_set = set()
for dt in datasets:
    dt_file = pd.read_csv(r"C:\Users\HEVAL\Desktop\GO_Prediction\datasets\final\{}".format(dt),sep="\t")
    missing_prot = dt_file.loc[~dt_file["Protein_Id"].isin(df["Protein_Id"])]
    print(len(dt_file)," - ",len(missing_prot))
    prot_set.update(set(dt_file["Protein_Id"]))

prot_df = pd.DataFrame({"Protein_Id":list(prot_set)})
    
transformer_missing_prot = prot_df.loc[~prot_df["Protein_Id"].isin(df["Protein_Id"])]
print(len(transformer_missing_prot))



