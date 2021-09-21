# PROBE (Protein RepresentatiOn BEnchmark): Function-Centric Evaluation of Protein Representation Methods

- PROBE runs benchmark analyses on protein representation/feature vectors of any representation learning method in order to evaluate its predictive performance on protein function related predictive tasks, and to and compare it other methods from literature.

- Aiming to evaluate how much each representation model captures different facets of functional information, we constructed and applied 4 independent benchmark tests based on;
  - inferring semantic similarities between proteins,
  - predicting ontology-based protein functions, 
  - classifying drug target proteins according to their families, and
  - estimating protein-protein binding affinities.

- PROBE is part of the the study entitled [Evaluation of Methods for Protein Representation Learning: A Quantitative Analysis](https://www.biorxiv.org/content/10.1101/2020.10.28.359828v1) which is schematically summarized in the figure below:<br/>
 
 ![Summary of The Study](https://github.com/serbulent/TrainableRepresentationAnalysis/blob/master/evalprotrep_summary_figure.jpg)


# How to Run PROBE (Protein RepresentatiOn BEnchmark)

- **Step-by-step operation:**
1. Clone this repository
2. Install dependencies (given below)
3. Download ['data' directory](https://drive.google.com/drive/folders/1N2TzFVSgdt2oZECmpTtpGHvJQMvza0i6?usp=sharing) place it under the directory 'data'.
4. If you wish to benchmark one or more of the 20 protein representation methods from the literature (that are included in our study) download [representation vector files for all human proteins](https://drive.google.com/drive/u/1/folders/1WmYyaBhOYtI4Hzbsg2sTQHRN6LVrYFhw) for benchmarks 1, 2 & 3, and download [representation vector files for the samples in the SKEMPI dataset](https://drive.google.com/drive/u/1/folders/18sVmR0Xx_QfmjeqCPxz3gS5DS09FqS_T) for benchmark 4, and place those csv formatted vector files directly under the directory 'data/representation_vectors'. If you wish to benchmark your own protein representation method, please prepare the vector files by following the steps provided under "Benchmarking your own representation model" below, and place csv formatted vector files directly under the directory 'data/representation_vectors'.
5. Edit the configuration file (example config file is provided below) by changing parameters as desired and setting paths of your file/files.
6. cd into the bin directory and run PROBE.py
 - i.e., python PROBE.py

- **Dependencies**
  - Python 3.8.1
  - Pandas 1.1.4
  - PyYaml 5.1
  - Scikit-Learn 0.22
  - Scikit-MultiLearn 0.2.0
  - Tqdm 4.51

- It is also possible to run PROBE online via [Code Ocean](https://codeocean.com/capsule/8584011/tree).

- **Example configuration file:**

```yaml
##Representation name (used for naming output files):
representation_name: AAC
#representation_name: LEARNED-VEC
#representation_name: T5

#Benchmarks (should be one of the "similarity","family","function","affinity","all"):
# "similarity" for running protein semantic similarity inference benchmark
# "function" for running ontology-based function prediction benchmark
# "family" for running drug target protein family classification benchmark
# "affinity" for running protein-protein binding affinity estimation benchmark
# "all" for running all benchmarks
benchmark: all

#Path of the file containing representation vectors of UniProtKB/Swiss-Prot human proteins:
representation_file_human: ../data/representation_vectors/AAC_UNIPROT_HUMAN.csv
#representation_file_human: ../data/representation_vectors/LEARNED-VEC_UNIPROT_HUMAN.csv
#representation_file_human: ../data/representation_vectors/T5_UNIPROT_HUMAN.csv

#Path of the file containing representation vectors of samples in the SKEMPI dataset: 
representation_file_affinity: ../data/representation_vectors/skempi_aac_representation_multi_col.csv
#representation_file_affinity: ../data/representation_vectors/skempi_learned-vec_representation_multi_col.csv
#representation_file_affinity: ../data/representation_vectors/skempi_t5_representation_multi_col.csv

#Semantic similarity inference benchmark dataset (should be a list that includes any combination of "Sparse", "200", and "500"):
similarity_tasks: ["Sparse","200","500"]

#Ontology-based function prediction benchmark dataset in terms of GO aspect (should be one of the following: "MF", "BP", "CC", or "All_Aspects"):
function_prediction_aspect: All_Aspects

#Ontology-based function prediction benchmark dataset in terms of size-based-splits (should be one of the following: "High", "Middle", "Low", or "All_Data_Sets")
function_prediction_dataset: All_Data_Sets

#Drug target protein family classification benchmark dataset in terms of similarity-based splits (should be a list that includes any combination of "nc", "uc50", "uc30", and "mm15")
family_prediction_dataset: ["nc","uc50","uc30","mm15"]

#Detailed results (can be True or False)
detailed_output: False

```

# Definition of output files

  - **Default output (these files are produced in the default run mode)**:

    - Semantic similarity prediction:

      - "Semantic_sim_inference_similarity_matrix_type_representation_name_.csv": This file includes semantic similarity correlation results for the selected representation method, on the selected dataset(s).
      - Similarity matrix type (dataset); 500: well-annotated 500 proteins, 200: well-annotated 200 proteins, sparse: sparse uniform dataset.

    - Ontology-based protein function prediction:

      - "Ontology_based_function_prediction_5cv_mean_representation_name.tsv": This file includes protein function prediction performance results (mean values of the 5-fold cross validation) for the selected representation method, on the selected dataset(s).
      - "Ontology_based_function_prediction_5cv_std_representation_name.tsv": This file includes standard deviation values for prediction performance scores on each fold of the 5-fold cross validation.
      - For detailed explanations regarding datasets, please see Methods sub-section entitled "Ontology-based Protein Function Prediction Benchmark" in our paper entitled: "Evaluation of Methods for Protein Representation Learning: A Quantitative Analysis".

    - Drug-target protein family classification:

      - "Drug_target_protein_family_classification_mean_results_dataset_name_representation_name.csv": This file includes the average overall drug target protein family classification performance results (mean of all protein family classes).
      - "Drug_target_protein_family_classification_class_based_results_dataset_name_representation_name.csv": This file includes family/class based drug target protein family classification performance results.
      - Dataset names (train-test split strategies are different between these datasets); nc: Non-clustred/Random split, uc50: Uniclust50 (50% sequence similarity-based split), uc30: Uniclust30 (30% sequence similarity-based split), mm15: MMSeq-15(15% sequence similarity-based split).

    - Protein-protein binding affinity estimation:

      - "Affinit_prediction_skempiv1_representation_name.csv": This file includes binding affinity estimation test result as mean scores of 10-fold cross-validation, in terms of mean squared error (MSE), mean absolute error (MAE), and Pearson correlation.
      - "Affinit_prediction_skempiv1_representation_name_detail.csv":This file includes binding affinity estimation test results of 10-fold cross-validation, independently calculated for each fold, in terms of mean squared error (MSE), mean absolute error (MAE), and Pearson correlation.

  - **Detailed output (these files are only produced when detailed_output parameter set as True):**

    - Semantic similarity prediction:

      - "Semantic_sim_inference_detailed_distance_scores_ontology_type_similarity_matrix_type.pkl": This file includes semantic similarity scores produced by the representation model.

    - Ontology-based protein function prediction:

      - "Ontology_based_function_prediction_dataset_name_representation_name_model.pkl":  This file includes scikit-learn SVM models trained for each dataset.
      - "Ontology_based_function_prediction_dataset_name_representation_name_predictions.tsv": This file includes predicted GO term labels for each dataset.
      - "Ontology_based_function_prediction_5cv_representation_name.tsv": This file includes prediction performance scores for each fold of the 5-fold cross-validation.

    - Drug-target protein family classification:

      - "Drug_target_protein_family_classification_score_type_dataset_name_representation_name.npy": This file includes individual scores (f1,accuray,mcc) for each fold of the the 10-fold cross-validation (overall).
      - "Drug_target_protein_family_classification_class_based_score_type_dataset_name_representation_name.npy": This file includes scores (f1,accuray,mcc) for each fold of the 10-fold cross-validation (per protein family).
      - "Drug_target_protein_family_classification_confusion_dataset_name_representation_name.csv": This file includes confusion matrices of each fold in the 10-fold cross-validation.
      - "Drug_target_protein_family_classification_class_based_support_dataset_name_representation_name.npy": This file includes support values for each class, per fold used in the 10-fold cross-validation.


# **Benchmarking your own representation model**

- Semantic similarity inference, Ontology-based protein function prediction and drug target protein family classification tasks can be run for any protein representation vector dataset. Similar to reproducing the analyses done in the study, there are two possible ways to do this: (i) running the tool on [Code Ocean](https://codeocean.com/capsule/858401), and (ii) cloning the Github repo and running locally (this option is advised if you plan to run additional tasks over the default ones, as the runtime may significantly increase).
  
  - Prepraration of the input vector dataset: 
    - Generate your representation vectors for all human proteins ([amino acid sequences of canonical isoform human proteins](https://drive.google.com/file/d/1wXF2lmj4ZTahMrl66QpYM2TvHmbcIL6b/view?usp=sharing)), and for the samples in the SKEMPI dataset [SKEMPI_seq.txt](https://drive.google.com/file/d/1m5jssC0RMsiFT_w-Ykh629Pw_An3PInI/view?usp=sharing).
  - Format of the protein representation file:
    - Each row corresponds to the representation vector of a particular protein.
    - Columns: first column's header one should be "Entry", and the rest of the column headers should contain the UniProt protein accessions of respective proteins (i.e., each column in this file corresponds to a different protein).
    - Rows: After column headers, the rows of the first column should contain the index number that correspond to dimensions of the vector, rows of other columns should contain representation vector values for the corresponding proteins (i.e. each row in this file corresponds to a dimension of representation vectors).
    - All representation vectors in a file should have the same size (i.e., fixed sized vectors).
  - Representation vectors of the whole dataset should be saved in a comma separated (csv) text file.
  - Example representation vector files can be found in the folder [representation_vectors_dataframes](https://drive.google.com/drive/u/1/folders/1B_TuRtz88Tv4R02WjliMXkbrJB5g5YXO).
  - The config file should be changed to provide the name of the new representation vector dataset, and to change other parameters as desired. 
  - Finally, the benchmark tests can be run (either on CodeOcean or locally by cloning the GitHub repo) as described above.


# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
