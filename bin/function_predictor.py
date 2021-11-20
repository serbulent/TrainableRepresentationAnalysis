# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing
import stratification as sf

from tqdm import tqdm
from datetime import datetime
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from sklearn.utils import shuffle

aspect_type = ""
dataset_type = ""
representation_dataframe = ""
representation_name = ""
detailed_output = False


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn


def check_and_drop_for_at_least_two_class_sample_exits(y):
    for column in y:
        column_sum = np.sum(y[column].array)
        if column_sum < 2:
            print('At least 2 positive samples are required for each class {0} class has {1} positive samples. Class {0} will be excluded !'
                  .format(column, column_sum))
            y = y.drop(columns=[column])
    return y


def MultiLabelSVC_cross_val_predict(representation_name, dataset, dt_merge, clf):

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
    prediction_list = []

    for random_state in range(10):
        dt_shuffled = shuffle(dt_merge, random_state=random_state)
        X = dt_shuffled['Vector']
        Xn = np.array(np.asarray(X.values.tolist()), dtype=float)
        y = dt_shuffled.iloc[:, 1:-2]
        # Find away for not cheking this every time
        y = check_and_drop_for_at_least_two_class_sample_exits(y)

        fold_index_list = sf.proba_mass_split(y.to_numpy())
        for fold_train_index, fold_test_index in sf.split(fold_index_list):
            train_X, test_X = Xn[fold_train_index], Xn[fold_test_index]
            train_Y, test_Y = y.iloc[fold_train_index], y.iloc[fold_test_index]
            #breakpoint()
            clf.fit(train_X, train_Y)
            if detailed_output:
                with open(r"../results/Ontology_based_function_prediction_{1}_{0}_model_shuffle_{2}.pkl".format(representation_name, dataset.split(".")[0], random_state), "wb") as file:
                    pickle.dump(clf, file)
            y_pred = clf.predict(test_X)
            prediction_list.append(y_pred)

            #breakpoint()
            acc = accuracy_score(test_Y, y_pred)
            acc_cv.append(np.round(acc, decimals=5))
            f1_mi = f1_score(test_Y, y_pred, average="micro")
            f1_mi_cv.append(np.round(f1_mi, decimals=5))
            f1_ma = f1_score(test_Y, y_pred, average="macro")
            f1_ma_cv.append(np.round(f1_ma, decimals=5))
            f1_we = f1_score(test_Y, y_pred, average="weighted")
            f1_we_cv.append(np.round(f1_we, decimals=5))
            pr_mi = precision_score(test_Y, y_pred, average="micro")
            pr_mi_cv.append(np.round(pr_mi, decimals=5))
            pr_ma = precision_score(test_Y, y_pred, average="macro")
            pr_ma_cv.append(np.round(pr_ma, decimals=5))
            pr_we = precision_score(test_Y, y_pred, average="weighted")
            pr_we_cv.append(np.round(pr_we, decimals=5))
            rc_mi = recall_score(test_Y, y_pred, average="micro")
            rc_mi_cv.append(np.round(rc_mi, decimals=5))
            rc_ma = recall_score(test_Y, y_pred, average="macro")
            rc_ma_cv.append(np.round(rc_ma, decimals=5))
            rc_we = recall_score(test_Y, y_pred, average="weighted")
            rc_we_cv.append(np.round(rc_we, decimals=5))
            hamm = hamming_loss(test_Y, y_pred)
            hamm_cv.append(np.round(hamm, decimals=5))

    means = list(np.mean([acc_cv, f1_mi_cv, f1_ma_cv, f1_we_cv, pr_mi_cv,
                          pr_ma_cv, pr_we_cv, rc_mi_cv, rc_ma_cv, rc_we_cv, hamm_cv], axis=1))
    means = [np.round(i, decimals=5) for i in means]

    stds = list(np.std([acc_cv, f1_mi_cv, f1_ma_cv, f1_we_cv, pr_mi_cv,
                        pr_ma_cv, pr_we_cv, rc_mi_cv, rc_ma_cv, rc_we_cv, hamm_cv], axis=1))
    stds = [np.round(i, decimals=5) for i in stds]
    
    #breakpoint()
    return ([representation_name+"_"+dataset, acc_cv, f1_mi_cv, f1_ma_cv, f1_we_cv, pr_mi_cv, pr_ma_cv, pr_we_cv, rc_mi_cv, rc_ma_cv, rc_we_cv, hamm_cv],
            [representation_name+"_"+dataset]+means,
            [representation_name+"_"+dataset]+stds,
            prediction_list)


def ProtDescModel():
    #desc_file = pd.read_csv(r"protein_representations\final\{0}_dim{1}.tsv".format(representation_name,desc_dim),sep="\t")
    datasets = os.listdir(r"../data/auxilary_input/GO_datasets")
    if dataset_type == "All_Data_Sets" and aspect_type == "All_Aspects":
        filtered_datasets = datasets
    elif dataset_type == "All_Data_Sets":
        filtered_datasets = [
            dataset for dataset in datasets if aspect_type in dataset]
    elif aspect_type == "All_Aspects":
        filtered_datasets = [
            dataset for dataset in datasets if dataset_type in dataset]
    else:
        filtered_datasets = [
            dataset for dataset in datasets if aspect_type in dataset and dataset_type in dataset]
    cv_results = []
    cv_mean_results = []
    cv_std_results = []

    for dt in tqdm(filtered_datasets, total=len(filtered_datasets)):
        print(r"Protein function prediction is started for the dataset: {0}".format(
            dt.split(".")[0]))
        dt_file = pd.read_csv(
            r"../data/auxilary_input/GO_datasets/{0}".format(dt), sep="\t")
        #if 'GO:0008511' in dt_file.columns:
            #dt_file.drop('GO:0008511', axis=1, inplace=True)

        dt_merge = dt_file.merge(
            representation_dataframe, left_on="Protein_Id", right_on="Entry")
        cpu_number = multiprocessing.cpu_count()
        model = MultiLabelSVC_cross_val_predict(representation_name, dt.split(
            ".")[0], dt_merge, clf=BinaryRelevance(SGDClassifier(n_jobs=cpu_number, random_state=42)))
        cv_results.append(model[0])
        cv_mean_results.append(model[1])
        cv_std_results.append(model[2])

        #predictions = dt_merge.iloc[:, :6]
        #breakpoint()
        #predictions["predicted_values"] = list(model[3].toarray())
        #if detailed_output:
        #    predictions.to_csv(r"../results/Ontology_based_function_prediction_{1}_{0}_predictions.tsv".format(
        #        representation_name, dt.split(".")[0]), sep="\t", index=None)

    return (cv_results, cv_mean_results, cv_std_results)

# def pred_output(representation_name, desc_dim):


def pred_output():
    model = ProtDescModel()
    cv_result = model[0]
    df_cv_result = pd.DataFrame({"Model": pd.Series([], dtype='str'), "Accuracy": pd.Series([], dtype='float'), "F1_Micro": pd.Series([], dtype='float'),
                                 "F1_Macro": pd.Series([], dtype='float'), "F1_Weighted": pd.Series([], dtype='float'), "Precision_Micro": pd.Series([], dtype='float'),
                                 "Precision_Macro": pd.Series([], dtype='float'), "Precision_Weighted": pd.Series([], dtype='float'), "Recall_Micro": pd.Series([], dtype='float'),
                                 "Recall_Macro": pd.Series([], dtype='float'), "Recall_Weighted": pd.Series([], dtype='float'), "Hamming_Distance": pd.Series([], dtype='float')})
    for i in cv_result:
        df_cv_result.loc[len(df_cv_result)] = i
    if detailed_output:
        df_cv_result.to_csv(r"../results/Ontology_based_function_prediction_5cv_{0}.tsv".format(
            representation_name), sep="\t", index=None)

    cv_mean_result = model[1]
    df_cv_mean_result = pd.DataFrame({"Model": pd.Series([], dtype='str'), "Accuracy": pd.Series([], dtype='float'), "F1_Micro": pd.Series([], dtype='float'),
                                      "F1_Macro": pd.Series([], dtype='float'), "F1_Weighted": pd.Series([], dtype='float'), "Precision_Micro": pd.Series([], dtype='float'),
                                      "Precision_Macro": pd.Series([], dtype='float'), "Precision_Weighted": pd.Series([], dtype='float'), "Recall_Micro": pd.Series([], dtype='float'),
                                      "Recall_Macro": pd.Series([], dtype='float'), "Recall_Weighted": pd.Series([], dtype='float'), "Hamming_Distance": pd.Series([], dtype='float')})

    # pd.DataFrame(columns=["Model","Accuracy","F1_Micro","F1_Macro","F1_Weighted","Precision_Micro","Precision_Macro","Precision_Weighted",\
    #                                     "Recall_Micro","Recall_Macro","Recall_Weighted","Hamming_Distance"])

    for j in cv_mean_result:
        df_cv_mean_result.loc[len(df_cv_mean_result)] = j
    df_cv_mean_result.to_csv(
        r"../results/Ontology_based_function_prediction_5cv_mean_{0}.tsv".format(representation_name), sep="\t", index=None)

# save std deviation of scores to file
    cv_std_result = model[2]
    df_cv_std_result = pd.DataFrame({"Model": pd.Series([], dtype='str'), "Accuracy": pd.Series([], dtype='float'), "F1_Micro": pd.Series([], dtype='float'),
                                     "F1_Macro": pd.Series([], dtype='float'), "F1_Weighted": pd.Series([], dtype='float'), "Precision_Micro": pd.Series([], dtype='float'),
                                     "Precision_Macro": pd.Series([], dtype='float'), "Precision_Weighted": pd.Series([], dtype='float'), "Recall_Micro": pd.Series([], dtype='float'),
                                     "Recall_Macro": pd.Series([], dtype='float'), "Recall_Weighted": pd.Series([], dtype='float'), "Hamming_Distance": pd.Series([], dtype='float')})

    # pd.DataFrame(columns=["Model","Accuracy","F1_Micro","F1_Macro","F1_Weighted","Precision_Micro","Precision_Macro","Precision_Weighted",\
    #                                     "Recall_Micro","Recall_Macro","Recall_Weighted","Hamming_Distance"])

    for k in cv_std_result:
        df_cv_std_result.loc[len(df_cv_std_result)] = k
    df_cv_std_result.to_csv(r"../results/Ontology_based_function_prediction_5cv_std_{0}.tsv".format(
        representation_name), sep="\t", index=None)


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
