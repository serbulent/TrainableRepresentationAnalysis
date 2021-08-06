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
from sklearn.model_selection import KFold
import os
from numpy import savetxt


def create_folds():

    protein_list = pd.read_csv('entry_class_n.csv')
    uniclust50 = pd.read_csv('uniclust50_2018_08/uniclust50_2018_08.tsv', sep='\t')
    uniclust50.columns =['first', 'second']

    train = pd.DataFrame()
    test = pd.DataFrame()

    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(protein_list):
        #print("TRAIN:", train_index, "TEST:", test_index)
        missing = 0
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        #print(protein_list.values.tolist().index(protein_list['Entry'][10]))
        for index in tqdm(test_index, total=len(test_index)):
            if(protein_list['Entry'][index] == "P0DTE5" or protein_list['Entry'][index] == "P0DTE4" or protein_list['Entry'][index] == "P0DTE8" or protein_list['Entry'][index] == "P0DTE7" or protein_list['Entry'][index] == "P0DUB6" or protein_list['Entry'][index] == "P0DTE0" or protein_list['Entry'][index] == "P0DSN6" or protein_list['Entry'][index] == "A0A096LPK9" or protein_list['Entry'][index] == "A0A0X1KG70"):
                train_index.remove(protein_list.index(protein_list['Entry'][index]))
                test_index.remove(protein_list.index(protein_list['Entry'][index]))
            else:
                try:
                    representative_protein_id = uniclust50[uniclust50['second'] == protein_list['Entry'][index]]['first'].item()
                    cluster = uniclust50[uniclust50['first'] == representative_protein_id]['second'].to_list()
                    cluster.remove(protein_list[index])
                except:
                    m = 2
        
                for clust in cluster:
                    try:
                          protein_index = protein_list.index(clust)
                          train_index.remove(protein_index)
    
                    except:
                          m = 1
        print(len(train_index))
        test = test.append(pd.Series(test_index), ignore_index = True)
        train = train.append(pd.Series(train_index), ignore_index = True)
    
    print(test)      
    print(train)

    train.to_csv("indexes/uniclust50_trainindex_n1.csv")
    test.to_csv("indexes/testindex_n1.csv")
            

create_folds()

