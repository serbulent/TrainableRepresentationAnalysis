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
from sklearn.metrics import accuracy_score


representation_name = ""
representation_path = ""

def score_protein_rep():
#def score_protein_rep(pkl_data_path):

    vecsize = 0
    protein_list = pd.read_csv('../data/auxilary_input/entry_class.csv')
    #protein_list = pd.read_csv('entry_class.csv')
    dataframe = pd.read_csv(representation_path)
    #dataframe = pd.read_pickle(pkl_data_path)
    vecsize = dataframe.shape[1]-1    
    x = np.empty([0, vecsize])
    y = []
    print("\n\nPreprocess data for drug-target protein family prediction ...\n ")
    for index, row in tqdm(protein_list.iterrows(), total=len(protein_list)):
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
 
    f1 = []
    accuracy = []
    mcc = []
    print('Calculating family predictions...\n')
    for i in tqdm(range(5)): 
        clf = linear_model.SGDClassifier(class_weight="balanced", loss="log", penalty="elasticnet", max_iter=1000, tol=1e-3,random_state=i,n_jobs=-1)
        clf2 = OneVsRestClassifier(clf,n_jobs=-1)
        y_pred = cross_val_predict(clf2, x, y, cv=10, n_jobs=-1)
        mcc.append(matthews_corrcoef(y, y_pred, sample_weight = y))
        f1_ = f1_score(y, y_pred, average='weighted')
        f1.append(f1_)
        ac = accuracy_score(y, y_pred)
        accuracy.append(ac)               
    
    report = pd.DataFrame()    
    f1mean = np.mean(f1, axis=0)
    #print(f1mean)
    f1mean = f1mean.round(decimals=5)
    f1std = np.std(f1).round(decimals=5)
    acmean = np.mean(accuracy, axis=0).round(decimals=5)
    acstd = np.std(accuracy).round(decimals=5)
    mccmean = np.mean(mcc, axis=0).round(decimals=5)
    mccstd = np.std(mcc).round(decimals=5)
    labels = ['Average Score', 'Standard Deviation']
    report['Protein Family'] = labels
    report['F1_score'] = [f1mean, f1std]
    report['Accuracy'] = [acmean, acstd]
    report['MCC'] = [mccmean, mccstd]
    report.to_csv('../results/dt_prot_famliy_pred_'+representation_name+'_report.csv',index=False)
    #report.to_csv('scores_general.csv')
    #print(report)   

#score_protein_rep("embedding_dataframes/SeqVec_dataframe_multi_col.pkl")

