# Evaluation of Methods for Protein Representation Learning
Evaluation of Methods for Protein Representation Learning: A Quantitative Analysisis an evalutation and review study on learned protein representations.

- The study is an investigation of the available learned protein representation methods.

- Aiming to evaluate how much each representation model captures different facets of functional information, we constructed and applied 4 independent benchmark tests based on;
  - inferring semantic similarities between proteins,
  - predicting ontology-based protein functions, 
  - classifying drug target proteins according to their families, and
  - estimating protein-protein binding affinities.

- The study is summarized in the figure below;<br/><br/> 
 
 ![Summary of The Study](https://github.com/serbulent/TrainableRepresentationAnalysis/blob/master/evalprotrep_summary_figure.jpg)

## Organization of this repository

- PROBE (Protein RepresentatiOn BEnchmark) is tool that can be used to evaluate any protein representation model. The tool can be found under 
**[bin](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/bin)** folder. Also study results can be reproduced easily using PROBE.

- **[paper_reproduction_scripts](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts)** folder includes codes used for benchmarks presented in the paper.

- **[function_prediction](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/function_prediction)** folder is under **Benchmark_Study** and includes codes used for "Ontology-based protein function prediction" task.

- **[semantic_similarity_inference](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/semantic_similarity_inference)** folder is under **Benchmark_Study** and includes codes used for "Semantic similarity inference" task.

- **[target_family_classification](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/target_family_classification)** folder is under **Benchmark_Study** and includes codes used for "Drug-target protein family classification" task.

- **[binding_affinity_estimation](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/binding_affinity_estimation)** folder is under **Benchmark_Study** and includes codes used for "Protein-protein binding affinity estimation" task.

- **[preprocess](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/preprocess)** folder is under **[paper_reproduction_scripts](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts)** and includes codes used for data preprocessing for "Ontology-based protein function prediction" and "Semantic similarity inference" tasks.

# Data Availability
- The large files used for the benchmark is shared on [GDrive](https://drive.google.com/drive/folders/1adgnOlb-4gQLgxEdsFmwtYFoYaiq5Eva) and the main directory structure is shown below.

The data needed by the PROBE tool is located under "PROBE" folder. The data belongs to benchmark study was located under the "Benchmark_IO_data" folder. Each task and  shared data has its own folder. The directory structure is self-explanatory and standard for all tasks, hence some of the folders might be empty based on task for now. 

Other than that the reusable 20 precalculated protein representation vectors for human proteins can be found under **"[Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/HUMAN_PROTEIN_VECTORS](https://drive.google.com/drive/u/1/folders/1WmYyaBhOYtI4Hzbsg2sTQHRN6LVrYFhw)"** folder for all tasks except protein-protein binding affinity estimation. The protein representation vectors belongs to protein-protein binding affinity estimation task is in **"[Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/SKEMPI](https://drive.google.com/drive/u/1/folders/18sVmR0Xx_QfmjeqCPxz3gS5DS09FqS_T)"** folder for further use.

<pre>
-PROBE 
-Benchmark_IO_data
|
|---GO_prediction
   |
   |---preprocess
   |---input_data
   |---results
|---Embedding_Similarity
   |
   |---preprocess
   |---input_data
   |---results
|---Protein_Family_Prediction
   |
   |---preprocess
   |---input_data
   |---results
|---Protein_Affinity_Prediction
   |
   |---preprocess
   |---input_data
   |---results
|---Shared_Data
</pre>

# **Benchmarking your own representation model**

- Semantic similarity inference, Ontology-based protein function prediction and drug target protein family classification tasks can be run for any protein representation vector dataset. Similar to reproducing the analyses done in the study, there are two possible ways to do this: (i) running the tool on [Code Ocean](https://codeocean.com/capsule/858401), and (ii) cloning the Github repo and running locally (this option is advised if you plan to run additional tasks over the default ones, as the runtime may significantly increase).
  
  - Prepraration of the input vector dataset: 
    - Generate your representation vectors for all human proteins (i.e. [canonical isoforms](https://drive.google.com/file/d/1wXF2lmj4ZTahMrl66QpYM2TvHmbcIL6b/view?usp=sharing))   Also SKEMPI dataset which can be found at [SKEMPI_seq.txt](https://drive.google.com/file/d/1m5jssC0RMsiFT_w-Ykh629Pw_An3PInI/view?usp=sharing) file.
- Format of the protein representation file:
      - Each row corresponds to the representation vector of a particular protein.
      - Columns: first column's header one should be "Entry", and the rest of the column headers should contain the UniProt protein accessions of respective proteins (i.e., each column in this file corresponds to a different protein).
      - Rows: After column headers, the rows of the first column should contain the index number that correspond to dimensions of the vector, rows of other columns should contain representation vector values for the corresponding proteins (i.e. each row in this file corresponds to a dimension of representation vectors).
      - All representation vectors in a file should have the same size (i.e., fixed sized vectors).

  - Representation vectors of the whole dataset should be saved in a comma separated (csv) text file.

  - Example representation vector files can be found in the folder [representation_vectors_dataframes](https://drive.google.com/drive/u/1/folders/1B_TuRtz88Tv4R02WjliMXkbrJB5g5YXO).

  - The config file should be changed to provide the name of the new representation vector dataset, and to change other parameters as desired.
  
  - Finally, the benchmark tests can be run (either on CodeOcean or locally by cloning the GitHub repo) as described above.


# How to Run PROBE (Protein RepresentatiOn BEnchmark)

- This tool runs benchmark analyses on the protein representation/feature vectors of different representation learning methods to evaluate and compare their predictive performance on protein function related predictive tasks.

- The tool can be run on [Code Ocean](https://codeocean.com/capsule/858401) or be clonned from this repository which can be found in generalized_representation_benchmark directory.

- **Dependencies**
  - Python 3.8.1
  - Pandas 1.1.4
  - PyYaml 5.1
  - Scikit-Learn 0.22
  - Scikit-MultiLearn 0.2.0
  - Tqdm 4.51 

- **Step-by-step operation:**
1. Clone this repository
2. Install dependencies
3. Download ['data' directory](https://drive.google.com/drive/folders/1N2TzFVSgdt2oZECmpTtpGHvJQMvza0i6?usp=sharing) place it in generalized_representation_benchmark directory with directory name 'data'.
4. Edit configuration file and set paths of your representation file/files.
5. Go to the PROBE directory and run the PROBE.py

 - i.e. cd PROBE && python PROBE.py

- **Example configuration file**
```yaml
#benchmark should be one of the "similarity","family","function","affinity","all"
# "similarity" for running protein semantic similarity inference benchmark
# "function" for running ontology-based function prediction benchmark
# "family" for running drug target protein family classification benchmark
# "affinity" for running drug target protein-protein binding affinity estimation benchmark
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
#similarity_tasks should be a list can be include any combination of "Sparse","200","500" 
#for selecting the semantic similarity inference benchmark dataset
similarity_tasks: ["Sparse","200","500"]
#function_prediction_aspect should be one of the "MF","BP","CC","All_Aspects" 
#for selecting the target Gene Ontology aspect/category for the protein function prediction benchmark
function_prediction_aspect: All_Aspects
#function_prediction_dataset should be one of the "High","Middle","Low","All_Data_Sets"
#for selecting the target size-based-split datasets for the protein function prediction benchmark
function_prediction_dataset: All_Data_Sets
#family_prediction_dataset a list can be include any combination of "nc", "uc50", "uc30", "uc15" 
#for selecting the target identity-based-similarity datasets protein-protein binding affinity estimation benchmark
family_prediction_dataset: ["nc","uc50","uc30","mm15"]
#detailed_output can be True or False
detailed_output: True

```

# Definition of output files

  - **Default output (these files are produced in the default run mode)**:
    - Semantic similarity prediction:
      - Semantic_sim_inference_similarity_matrix_type_representation_name_.csv: This file includes semantic similarity correlation results for the selected representation method, on the selected dataset(s).
      - Similarity matrix type; 500: well-annotated 500 proteins, 200: well-annotated 200 proteins, sparse: sparse uniform dataset.
    - Ontology-based protein function prediction:
      - Ontology_based_function_prediction_5cv_mean_representation_name.tsv: This file includes protein function prediction performance results (mean values of the 5-fold cross validation) for the selected representation method, on the selected dataset(s).
      - Ontology_based_function_prediction_5cv_std_representation_name.tsv: This file includes standard deviation values for prediction performance scores on each fold of the 5-fold cross validation.
      - For detailed explanations regarding datasets, please see Methods sub-section entitled "Ontology-based Protein Function Prediction Benchmark" in our paper.
    - Drug-target protein family classification:
      - Drug_target_protein_family_classification_mean_results_dataset_name_representation_name.csv: This file includes the average overall drug target protein family classification performance results (mean of all protein family classes).
      - Drug_target_protein_family_classification_class_based_results_dataset_name_representation_name.csv: This file includes family/class based drug target protein family classification performance results.
      - Dataset names (train-test split strategies are different between these datasets); nc: Non-clustred/Random split, uc50: Uniclust50 (50% sequence similarity-based split), uc30: Uniclust30 (30% sequence similarity-based split), mm15: MMSeq-15(15% sequence similarity-based split).
    - Protein-protein binding affinity estimation:
      - Affinit_prediction_skempiv1_representation_name.csv: This file includes binding affinity estimation test result as mean scores of 10-fold cross-validation, in terms of mean squared error (MSE), mean absolute error (MAE), and Pearson correlation.
      - Affinit_prediction_skempiv1_representation_name_detail.csv:This file includes binding affinity estimation test results of 10-fold cross-validation, independently calculated for each fold, in terms of mean squared error (MSE), mean absolute error (MAE), and Pearson correlation.

  - **Detailed output (these files are only produced when detailed_output parameter set as True):**
    - Semantic similarity prediction:
      - Semantic_sim_inference_detailed_distance_scores_ontology_type_similarity_matrix_type.pkl: This file includes the semantic similarity scores produced by the representation model.
    - Ontology-based protein function prediction:
      - Ontology_based_function_prediction_dataset_name_representation_name_model.pkl:  This file includes scikit-learn SVM models trained for each dataset.
      - Ontology_based_function_prediction_dataset_name_representation_name_predictions.tsv: This file includes predicted GO term labels for each dataset.
      - Ontology_based_function_prediction_5cv_representation_name.tsv: This file includes prediction performance scores for each fold of the 5-fold cross-validation.
    - Drug-target protein family classification:
      - Drug_target_protein_family_classification_score_type_dataset_name_representation_name.npy: This file includes individual scores (f1,accuray,mcc) for each fold of the the 10-fold cross-validation (overall).
      - Drug_target_protein_family_classification_class_based_score_type_dataset_name_representation_name.npy: This file includes scores (f1,accuray,mcc) for each fold of the 10-fold cross-validation (per protein family).
      - Drug_target_protein_family_classification_confusion_dataset_name_representation_name.csv: This file includes confusion matrices of each fold in the 10-fold cross-validation.
      - Drug_target_protein_family_classification_class_based_support_dataset_name_representation_name.npy: This file includes support values for each class, per fold used in the 10-fold cross-validation.
  
# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
