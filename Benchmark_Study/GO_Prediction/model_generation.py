# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss

print(datetime.now())      
        
def MultiLabelSVC_cross_val_predict(descriptor, dataset, X, y, classifier):
    #dataset split, estimator, cv
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = classifier 
    y_pred = cross_val_predict(clf, X, y, cv=kf)
    clf.fit(X, y)

    with open(r"models\{0}\{0}_{1}.pkl".format(descriptor,dataset.split(".")[0]),"wb") as file:
        pickle.dump(clf,file)
        
    acc_cv = []
    f1_mi_cv = []
    f1_ma_cv = []
    f1_we_cv = []
    pr_mi_cv = []
    pr_ma_cv = []
    pr_we_cv = []
    rc_mi_cv = []
    rc_ma_cv = []
    rc_we_cv = []
    hamm_cv = []
    for fold_train_index,fold_test_index in kf.split(X,y):
        acc = accuracy_score(y.iloc[fold_test_index,:],y_pred[fold_test_index])
        acc_cv.append(acc)
        f1_mi = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        f1_mi_cv.append(f1_mi)
        f1_ma = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        f1_ma_cv.append(f1_ma)
        f1_we = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        f1_we_cv.append(f1_we)
        pr_mi = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        pr_mi_cv.append(pr_mi)
        pr_ma = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        pr_ma_cv.append(pr_ma)
        pr_we = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        pr_we_cv.append(pr_we)
        rc_mi = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        rc_mi_cv.append(rc_mi)
        rc_ma = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        rc_ma_cv.append(rc_ma)
        rc_we = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        rc_we_cv.append(rc_we)
        hamm = hamming_loss(y.iloc[fold_test_index,:],y_pred[fold_test_index])
        hamm_cv.append(hamm)

    return ([descriptor+"_"+dataset,acc_cv,f1_mi_cv,f1_ma_cv,f1_we_cv,pr_mi_cv,pr_ma_cv,pr_we_cv,rc_mi_cv,rc_ma_cv,rc_we_cv,hamm_cv],\
            [descriptor+"_"+dataset]+list(np.mean([acc_cv,f1_mi_cv,f1_ma_cv,f1_we_cv,pr_mi_cv,pr_ma_cv,pr_we_cv,rc_mi_cv,rc_ma_cv,rc_we_cv,hamm_cv], axis=1)),\
            y_pred)
   

def ProtDescModel(descriptor, desc_dim):   
    desc_file = pd.read_csv(r"protein_representations\final\{0}_dim{1}.tsv".format(descriptor,desc_dim),sep="\t")    
    datasets = os.listdir(r"datasets\final") 

    cv_results = []
    cv_mean_results = []
    for dt in datasets:
        dt_file = pd.read_csv(r"datasets\final\{0}".format(dt),sep="\t")
        dt_merge = dt_file.merge(desc_file,on="Protein_Id")

        dt_X = dt_merge.iloc[:,len(dt_file.columns)+1:]
        dt_y = dt_merge.iloc[:,1:len(dt_file.columns)]
        print("raw dt vs. dt_merge: {} - {}".format(len(dt_file),len(dt_merge)))

        model = MultiLabelSVC_cross_val_predict(descriptor, dt.split(".")[0], dt_X, dt_y, classifier=BinaryRelevance(SVC(kernel="linear", random_state=42)))
        cv_results.append(model[0])                
        cv_mean_results.append(model[1])
        
        predictions = dt_merge.iloc[:,:len(dt_file.columns)]
        predictions["predicted_values"] = list(model[2].toarray())
        predictions.to_csv(r"predictions\{0}\{0}_{1}_predictions.tsv".format(descriptor,dt.split(".")[0]),sep="\t",index=None)

    return (cv_results, cv_mean_results)             


def pred_output(descriptor, desc_dim):
    model = ProtDescModel(descriptor=descriptor, desc_dim=desc_dim)
    cv_result = model[0]
    df_cv_result = pd.DataFrame(columns=["model","acc","f1_mi","f1_ma","f1_we","pr_mi","pr_ma","pr_we",\
                                         "rc_mi","rc_ma","rc_we","hamm"])
    for i in cv_result:
        df_cv_result.loc[len(df_cv_result)] = i
    df_cv_result.to_csv(r"results\{0}_5cv.tsv".format(descriptor),sep="\t",index=None)

    cv_mean_result = model[1]
    df_cv_mean_result = pd.DataFrame(columns=["model","acc","f1_mi","f1_ma","f1_we","pr_mi","pr_ma","pr_we",\
                                              "rc_mi","rc_ma","rc_we","hamm"])
    for j in cv_mean_result:
        df_cv_mean_result.loc[len(df_cv_mean_result)] = j
    df_cv_mean_result.to_csv(r"results\{0}_5cv_mean.tsv".format(descriptor),sep="\t",index=None)

print(datetime.now())      


# tcga = pred_output("tcga","50") 
# protvec = pred_output("protvec","100")  
# unirep = pred_output("unirep","5700")  
# gene2vec = pred_output("gene2vec","200")   
# learned_embed = pred_output("learned_embed","64") 
# mut2vec = pred_output("mut2vec","300")    
# seqvec = pred_output("seqvec","1024") 

#bepler = pred_output("bepler","100") 
# resnet_rescaled = pred_output("resnet-rescaled","256") 
# transformer_avg = pred_output("transformer","768") 
# transformer_pool = pred_output("transformer-pool","768") 

# apaac = pred_output("apaac","80") 
#ksep = pred_output("ksep","400") 








