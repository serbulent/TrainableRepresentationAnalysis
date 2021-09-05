# Evaluation of Methods for Protein Representation Learning
Evaluation of Methods for Protein Representation Learning: A Quantitative Analysisis an evalutation and review study on learned protein representations.

- The study is an investigation of the available learned protein representation methods.

- Aiming to evaluate how much each representation model captures different facets of functional information, we constructed and applied benchmarks based on;
  - The detection of semantic similarities between proteins (a.k.a Semantic similarity inference)
  - Ontology-based protein function prediction
  - Drug-target protein family classification
  - Protein-protein binding affinity estimation

- The study is summarized in the figure below;<br/><br/> 
 
 ![Summary of The Study](https://github.com/serbulent/TrainableRepresentationAnalysis/blob/master/evalprotrep_summary_figure.jpg)

## Organization of this repository

- PROBE (Protein RepresentatiOn BEnchmark) is tool that can be used to evaluate any protein representation model. The tool can be found under 
**generalized_representation_benchmark/PROBE** folder. Also study results can be reproduced easily using PROBE.

- **Benchmark_Study** folder includes codes used for benchmarks presented in the paper.

- **go_prediction** folder is under **Benchmark_Study** and includes codes for generalized visualization of the results.

- **function_prediction** folder is under **Benchmark_Study** and includes codes used for "Ontology-based protein function prediction" task.

- **semantic_similarity_inference** folder is under **Benchmark_Study** and includes codes used for "Semantic similarity inference" task.

- **target_family_classification** folder is under **Benchmark_Study** and includes codes used for "Drug-target protein family classification" task.

- **binding_affinity_estimation** folder is under **Benchmark_Study** and includes codes used for "Protein-protein binding affinity estimation" task.

- **data/preprocess** folder is under **Benchmark_Study** and includes codes used for data preprocessing for "Ontology-based protein function prediction" and "Semantic similarity inference" tasks.

# Data Availability
- The large files used for the benchmark is shared on [GDrive](https://drive.google.com/drive/folders/1adgnOlb-4gQLgxEdsFmwtYFoYaiq5Eva) and the main directory structure is shown below.

The data needed by the PROBE tool is located under "PROBE" folder. The data belongs to benchmark study was located under the "Benchmark_IO_data" folder. Each task and  shared data has its own folder. The directory structure is self-explanatory and standard for all tasks, hence some of the folders might be empty based on task for now. 

Other than that the reusable 20 precalculated protein representation vectors for human proteins can be found under **"Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/HUMAN_PROTEIN_VECTORS"** folder for all tasks except protein-protein binding affinity estimation. The protein representation vectors belongs to protein-protein binding affinity estimation task is in **"Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/SKEMPI"** folder for further use.

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

  - Semantic similarity prediction, Ontology-based protein function prediction, Drug target protein family classification and Protein-protein binding affinity estimation tasks can be run for any representation vector dataset. There are two possible ways to do this:
    - Cloning the capsule and running it on [Code Ocean](https://codeocean.com/capsule/858401)
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
  - Following the generation of the representation vector file, the benchmark tests can be run as described in the "How to Run PROBE" section.

# How to Run PROBE (Protein RepresentatiOn BEnchmark)

- This tool runs benchmark analyses on the protein representation vectors of different representation learning methods to evaluate and compare their predictive performance on protein function related predictive tasks.

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
family_prediction_dataset: ["nc","uc50","uc30","uc15"]
#detailed_output can be True or False
detailed_output: True

```

# Definition of output files

  - Default output (files that are produced in the default mode run):
    - Semantic similarity prediction:
      - Semantic_sim_pred_representation_name_similarity_matrix_type.csv: This file includes semantic similarity correlation results for the selected representation method, on the selected dataset.
      - Dataset names; 500: well-annotated 500 proteins, 200: well-annotated 200 proteins, sparse: sparse uniform dataset
    - Ontology-based protein function prediction:
      - Ontology_based_function_prediction_representation_name_5cv_mean.tsv: This file includes ontology-based function protein prediction performance results (mean values of the 5-fold cross validation) for the selected representation method, on the selected dataset(s).
      - Ontology_based_function_prediction_representation_name_5cv_std.tsv: This file includes standrt deviation values for prediction performance scores on each fold of the 5-fold cross validation.
      - For explanations regarding datasets, please see Methods sub-section entitled "Ontology-based Protein Function Prediction Benchmark" in out paper
    - Drug-target protein family classification:
      - family_classification_representation_name_mean_results_dataset_name.csv: This file includes the average drug target protein family classification performance results (mean of all protein family classes).
      - family_classification_representation_class_based_results_dataset_name.csv: This file includes family/class based drug target protein family classification performance results.
      - family_classification_confusion_representation_name_dataset_name.csv: This file includes confusion matrices of each fold applied during cross-validation.
      - Dataset names; nc: Non-clustred/Random, uc50: Uniclust50, uc30: Uniclust30, mm15: MMSeq-15
    - Protein-protein binding affinity estimation:
      - Affinit_prediction_skempiv1_representation_name.csv: This file includes affinity estimation test result as mean scores of 10-fold cross-validation which are mean squared error, mean absolute error, and correlation.
      - Affinit_prediction_skempiv1_representation_name_detail.csv:This file includes affinity estimation test results of 10-fold cross-validation for each fold which are mean squared error, mean absolute error, and correlation.

  - **Detailed output (these files are only produced when detailed_output parameter set as True):**
    - Semantic similarity prediction:
      - Detailed_Semantic_sim_pred_representation_name_dataset_name_ontology_type.pkl: This file includes the semantic similarity score the representation model. 
    - Ontology-based protein function prediction:
      - representation_name_dataset_name_model.pkl: This file includes scikit-learn SVM models trained for each dataset for the selected representation method.
      - representation_name_dataset_name_predictions.tsv: This file includes predicted GO term labels for each dataset.
      - Ontology_based_function_prediction_representation_name_5cv.tsv: This file includes prediction performance scores for each fold of the 5-fold cross-validation.
    - Drug-target protein family classification:
      - drug_target_family_pred_score_type_representation_name_dataset_name.npy: This file includes scores (f1,accuray,mcc) belngs to 10-fold cross-validation.
      - drug_target_family_pred_score_type_perclass_representation_name_dataset_name.npy: This file includes scores (f1,accuray,mcc) belngs to 10-fold cross-validation for each class.
  

# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
