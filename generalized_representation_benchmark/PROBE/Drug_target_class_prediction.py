# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:32:26 2020

@author: Muammer
"""

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd 
from numpy import save
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from decimal import *
getcontext().prec = 5


representation_name = ""
representation_path = ""
detailed_output = False

def score_protein_rep():

    vecsize = 0
    protein_list = pd.read_csv('../data/auxilary_input/entry_class.csv')
    dataframe = pd.read_csv(representation_path)
    vecsize = dataframe.shape[1]-1    
    x = np.empty([0, vecsize])
    y = []
    print("\n\nPreprocess data for drug target protein family prediction (class based)...\n")
    for index, row in tqdm(protein_list.iterrows(), total=len(protein_list)) :
        pdrow = dataframe.loc[dataframe['Entry'] == row['Entry']]
        if len(pdrow) != 0:
            a = pdrow.loc[ : , pdrow.columns != 'Entry']
            a = np.array(a)
            a.shape = (1,vecsize)
            x = np.append(x, a, axis=0)
            y.append(row['Class'])

    x = x.astype(np.float64)
    y = np.array(y)
    y = y.astype(np.float64)
    
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted', 'accuracy']
    target_names = ['Enzyme', 'Membrane receptor', 'Transcription factor', 'Ion channel', 'Other']
    labels = [1.0, 11.0, 12.0, 1005.0, 2000.0]
    
    f1_perclass = []
    accurac_perclass = []
    mcc_perclass = []
    mcc1 = []
    mcc11 = []
    mcc12 = []
    mcc1005 = []
    mcc2000 = []
    
    print('Calculating family predictions... (class based)\n')
    for i in tqdm(range(100)):
        y1 = np.empty([0])
        y11 = np.empty([0])
        y12 = np.empty([0])
        y1005 = np.empty([0])
        y2000 = np.empty([0])
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y,random_state=i)
        testsize = np.size(X_test,0)
        
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)
        
        for i in range(testsize):
            if y_test[i] == 1:
                y1 = np.append(y1 ,y_test[i])
            else:
                y1 = np.append(y1 ,0)
            if y_test[i] == 11:
                y11 = np.append(y11 ,y_test[i])
            else:
                y11 = np.append(y11 ,0)
            if y_test[i] == 12:
                y12 = np.append(y12 ,y_test[i])
            else:
                y12 = np.append(y12 ,0)
            if y_test[i] == 1005:
                y1005 = np.append(y1005 ,y_test[i])
            else:
                y1005 = np.append(y1005 ,0)
            if y_test[i] == 2000:
                y2000 = np.append(y2000 ,y_test[i])
            else:
                 y2000 = np.append(y2000 ,0)
        
        clf = linear_model.SGDClassifier(class_weight="balanced", loss="log", penalty="elasticnet", max_iter=1000, tol=1e-3,n_jobs=-1,random_state=i)
        clf2 = OneVsRestClassifier(clf,n_jobs=-1).fit(X_train, y_train)
        y_pred = clf2.predict(X_test)

        f1 = f1_score(y_test, y_pred, labels=labels, average=None)
        f1_perclass.append(f1.round(decimals=2))
        
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = cm.diagonal()
        accurac_perclass.append(accuracy.round(decimals=2))
        
        pred_data = clf2.predict(X_test)
        
        mcc1.append(matthews_corrcoef(y1, pred_data))
        mcc11.append(matthews_corrcoef(y11, pred_data))
        mcc12.append(matthews_corrcoef(y12, pred_data))
        mcc1005.append(matthews_corrcoef(y1005, pred_data))
        mcc2000.append(matthews_corrcoef(y2000, pred_data))
        
    mcc1 = np.array(mcc1)
    mcc11 = np.array(mcc11)
    mcc12 = np.array(mcc12)
    mcc1005 = np.array(mcc1005)
    mcc2000 = np.array(mcc2000)
    
    mcc_perclass.append(mcc1.round(decimals=2))
    mcc_perclass.append(mcc11.round(decimals=2))
    mcc_perclass.append(mcc12.round(decimals=2))
    mcc_perclass.append(mcc1005.round(decimals=2))
    mcc_perclass.append(mcc2000.round(decimals=2))
    
    #report = pd.DataFrame(result).transpose()
    report = pd.DataFrame()
    f1means = np.mean(f1_perclass, axis=0)
    acmeans = np.mean(accurac_perclass, axis=0)
    labels = ['Enzyme', 'Membrane receptor', 'Transcription factor', 'Ion channel', 'Other', 'Weighted Average']
    f1s = [f1means[0], f1means[1], f1means[2], f1means[3], f1means[4], np.average(f1means, weights=[len(y1), len(y11), len(y12), len(y1005), len(y2000)])]
    mccs = [mcc1.mean(), mcc11.mean(), mcc12.mean(), mcc1005.mean(), mcc2000.mean(), np.average([mcc1.mean(),  mcc11.mean(), mcc12.mean(), mcc1005.mean(), mcc2000.mean()], weights=[len(y1), len(y11), len(y12), len(y1005), len(y2000)])]
    accuracys = [acmeans[0], acmeans[1], acmeans[2], acmeans[3], acmeans[4], np.average(acmeans, weights=[len(y1), len(y11), len(y12), len(y1005), len(y2000)])]
    report['Families'] = labels
    report['F1_Score'] = [i.round(decimals=5) for i in f1s]
    report['Accuracy'] = [i.round(decimals=5) for i in accuracys]
    report['MCC'] = [i.round(decimals=5) for i in  mccs]
    report.to_csv('../results/dt_prot_famliy_pred_'+representation_name+'_report_class_based.csv',index=False)
    
    #print(report)
    if detailed_output:
        save('../results/drug_target_family_pred_f1_perclass_'+ representation_name +'.npy', f1_perclass)
        save('../results/drug_target_family_pred_accuracy_perclass_'+ representation_name +'.npy', accurac_perclass)
        save('../results/drug_target_family_pred_mcc_perclass_'+ representation_name +'.npy', mcc_perclass) 
        #save('../results/drug_target_family_pred_y_pred_'+ representation_name +'.npy', y_pred) 

#score_protein_rep("embedding_dataframes/SeqVec_dataframe_multi_col.pkl")


