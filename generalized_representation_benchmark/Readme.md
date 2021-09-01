# ProtBench: A Benchmarking Platform for Function-Centric Evaluation of Protein Representation Methods

- The tool can be run on [Code Ocean](https://codeocean.com/capsule/858401) or be clonned from this repository.

- For running this tool locally please download ['data' directory](https://drive.google.com/drive/folders/1N2TzFVSgdt2oZECmpTtpGHvJQMvza0i6?usp=sharing) and place it into the same folder in generalized_representation_benchmark directory.

- Aiming to evaluate how much each representation model captures different facets of functional information, we constructed and applied benchmark tests based on;
  - inferring semantic similarities between proteins,
  - predicting ontology-based protein functions, 
  - classifying drug target proteins according to their families, and
  - protein-protein binding affinity estimation.

- This tool runs benchmark analyses on the protein representation vectors of different representation learning methods
 to evaluate and compare their predictive performance on protein function related predictive tasks.

- The benchmark module run for all tests on one protein represenation method (e.g., AAC). The options can be set using probe_config.yaml and test can be run from linux console.

 - i.e. python PROBE.py

- **Example configuration file**
```yaml
#benchmark should be one of the "similarity","family","function","affinity","all"
# "similarity" for running protein semantic similarity inference benchmark
# "function" for running ontology-based function prediction benchmark
# "family" for running drug target protein family classification benchmark
# "affinity" for running drug target protein-protein interaction affinity estimation benchmark
# "all" for running all benchmarks
benchmark: all
#Path of the representation file for representation vector file to be benchmarked "similarity","family","function" tasks.
#The file should be consist of multiple columns. The first column should be named 'ENTRY' which includes UNIPROT ID, 
#and various subsequent columns corresponding to representation vector dimensions should exist.
representation_file_human: path_to_representation_file_human/AAC_UNIPROT_HUMAN.csv
#Path of the representation file for affinity task. 
#The file should be consist of multiple columns. The first column should be named 'ENTRY' which includes UNIPROT ID, 
#and various subsequent columns corresponding to representation vector dimensions should exist.
representation_file_affinity: path_to_representation_file_affinity/skempi_AAC.csv
#Representation name which is used to name output files
representation_name: AAC
#similarity_tasks should be a list can be include any combination of "Sparse","200","500" for selecting the semantic similarity inference benchmark dataset
similarity_tasks: ["Sparse","200","500"]
#function_prediction_aspect should be one of the "MF","BP","CC","All_Aspects" for selecting the target Gene Ontology aspect/category for the protein function prediction benchmark
function_prediction_aspect: All_Aspects
#function_prediction_dataset should be one of the "High","Middle","Low","All_Data_Sets" for selecting the target size-based-split datasets for the protein function prediction benchmark
function_prediction_dataset: All_Data_Sets
#family_prediction_dataset a list can be include any combination of "nc", "uc50", "uc30", "uc15" for selecting the target identity-based-similarity datasets protein-protein interaction affinity estimation benchmark
family_prediction_dataset: ["nc","uc50","uc30","uc15"]
#detailed_output can be True or False
detailed_output: True

```

- **Definition of output files:**

  - Default output (files that are produced in the default mode run):
    - Semantic similarity prediction:
      - Semantic_sim_pred_representation_name_similarity_matrix_type.csv: This file includes semantic similarity correlation results for the selected representation method, on the selected dataset.
    - Ontology-based protein function prediction:
      - Ontology_based_function_prediction_representation_name_5cv_mean.tsv: This file includes ontology-based function protein prediction performance results (mean values of the 5-fold cross validation) for the selected representation method, on the selected dataset(s).
      - Ontology_based_function_prediction_representation_name_5cv_std.tsv: This file includes standrt deviation values for prediction performance scores on each fold of the 5-fold cross validation.
    - Drug-target protein family classification:
       - family_classification_representation_name_mean_results_dataset_name.csv: This file includes the average drug target protein family classification performance results (mean of all protein family classes).
      - family_classification_representation_class_based_results_dataset_name.csv: This file includes family/class based drug target protein family classification performance results.
      - family_classification_confusion_representation_name_dataset_name.csv: This file includes confusion matrices of each fold applied during cross-validation.
    - Protein-protein binding affinity estimation:
      - Affinit_prediction_skempiv1_representation_name.csv: This file includes affinity estimation test result as mean scores of 10-fold cross-validation which are mean squared error, mean absolute error, and correlation.
      - Affinit_prediction_skempiv1_representation_name_detail.csv:This file includes affinity estimation test results of 10-fold cross-validation for each fold which are mean squared error, mean absolute error, and correlation.

  - Detailed output (these files are only produced when detailed_output parameter set as True):
    - Semantic similarity prediction:
      - Detailed_Semantic_sim_pred_representation_name_dataset_name_ontology_type.pkl: This file includes the semantic similarity score the representation model. 
    - Ontology-based protein function prediction:
      - representation_name_dataset_name_model.pkl: This file includes scikit-learn SVM models trained for each dataset for the selected representation method.
      - representation_name_dataset_name_predictions.tsv: This file includes predicted GO term labels for each dataset.
      - Ontology_based_function_prediction_representation_name_5cv.tsv: This file includes prediction performance scores for each fold of the 5-fold cross-validation.
    - Drug-target protein family classification:
      - drug_target_family_pred_score_type_representation_name_dataset_name.npy: This file includes scores (f1,accuray,mcc) belngs to 10-fold cross-validation.
      - drug_target_family_pred_score_type_perclass_representation_name_dataset_name.npy: This file includes scores (f1,accuray,mcc) belngs to 10-fold cross-validation for each class.
  
- **Benchmarking your own representation model:**

  - Semantic similarity prediction, Ontology-based protein function prediction and Drug target protein family classification tasks can be run for any representation vector dataset. There are two possible ways to do this:
    - Cloning the capsule and running it on Code Ocean. 
    - Clonning this repository  (this option is advised if you plan to run additional tasks over the default ones, as the run time may be elevated).
  
  - Prepraration of the input vector dataset: 
    - Generate your representation vectors for all human proteins (i.e. canonical isoforms) and SKEMPI dataset which can be found at [SKEMPI_seq.txt](https://drive.google.com/file/d/1m5jssC0RMsiFT_w-Ykh629Pw_An3PInI/view?usp=sharing) file.
    - File format:
      - Each row corresponds to the representation vector of a particular protein.
      - The first column header should be "Entry" and the rows contain the respective UniProt protein accessions.
      - Following column headers can be the index number of the dimension of the vector, the rows contain representation vectors' values (i.e. each column corresponds to a dimension of the representation vector).
      - Representation vectors should have a fixed size.
      - Save your representation vectors in comma separated (csv) text file.
      - All representation vectors files used in this study can be found in the folder [representation_vectors_dataframes](https://drive.google.com/drive/u/1/folders/1B_TuRtz88Tv4R02WjliMXkbrJB5g5YXO).
  - Following the generation of the representation vector file, the benchmark tests can be run as described above.
  - For the local run, dependencies can be found in the environment section of the code ocean capsule.


# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
