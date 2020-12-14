# ProtBench: A Benchmarking Platform for Function-Centric Evaluation of Protein Representation Methods

- The tool can be run on [Code Ocean](https://codeocean.com/capsule/858401) or be clonned from this repository.

- Aiming to evaluate how much each representation model captures different facets of functional information, we constructed and applied benchmark tests based on;
  - inferring semantic similarities between proteins,
  - predicting ontology-based protein functions, and
  - classifying drug target proteins according to their families.

- This tool runs benchmark analyses on the protein representation vectors of different representation learning methods
 to evaluate and compare their predictive performance on protein function related predictive tasks.

- The benchmark module will only run for fundamental tests on one
protein represenation method (e.g., SeqVec) in the default mode; however, all/selected tests can be run on 
all/selected representation methods using the options given below.
 - Also each task can be run independently by using -s , -f, -fam options solely.

- Individual benchmarks can be run independently by using -s , -f, -fam options solely. Different representation models can be
run by specifying the input vector file with the "-rf" parameter and the representation name with the "-rn" parameter.
  - i.e. python run_benchmarks.py -s -fam -f -rf ../data/representation_vectors/SeqVec_dataframe_multi_col.csv -rn SeqVec

- **Input arguments:**

  "-s","--similarity" (for running protein semantic similarity inference benchmark) <br>
  "-f", "--function_prediction" (for running ontology-based function prediction benchmark)<br>
  "-fam", "--family_prediction" (for running drug target protein family classification benchmark)<br>
  "-a", "--all" (for running all benchmarks)<br>
  "-d","--detailed_output", (for saving all results including detailed performance results and predictions)<br>
  "-sts","--similarity_tasks" , choices= ['Sparse','200','500','All'], default='Sparse' (for selecting the semantic
   similarity inference benchmark dataset)<br>
  "-fpa","--function_prediction_aspect", choices= ['MF','BP','CC','All_Aspects'], default='All_Aspects' (for selecting the
   target Gene Ontology aspect/category for the protein function prediction benchmark)<br>
  "-fpd","--function_prediction_dataset", choices= ['High','Middle','Low','All_Data_Sets'], default='Low' (for selecting the
   target size-based-split datasets for the protein function prediction benchmark)<br>
  "-rf", "--representation_file", (representation vector file to be benchmarked, it should be a pandas dataframe in pickle
   format which includes 'Entry' and 'Vector' columns)<br>
  "-rn", "--representation_name" (the name of the representation method to be analyzed)<br>

- **Definition of output files:**

  - Default output (files that are produced in the default mode run):
    - Semantic similarity prediction:
      - Semantic_sim_pred_representation_name_similarity_matrix_type.csv: This file includes semantic similarity correlation results for the selected representation method, on the selected dataset.
    - Ontology-based protein function prediction:
      - ontology_based_function_prediction_representation_name_5cv_mean.tsv: This file includes ontology-based function protein prediction performance results (mean values of the 5-fold cross validation) for the selected representation method, on the selected dataset(s).
    - Drug-target protein family classification:
      - dt_prot_famliy_pred_representation_name_report.csv: This file includes the average drug target protein family classification performance results (mean of all protein family classes).
      - dt_prot_famliy_pred_representation_name_report_class_based.csv: This file includes family/class based drug target protein family classification performance results.

  - Detailed output (these files are only produced when "--detailed_output" parameter is given):
    - Semantic similarity prediction:
      - None
    - Ontology-based protein function prediction:
      - representation_name_dataset_name_model.pkl: This file includes scikit-learn SVM models trained for each dataset for the selected representation method.
      - representation_name_dataset_name_predictions.tsv: This file includes predicted GO term labels for each dataset.
      - Ontology_based_function_prediction_representation_name_5cv.tsv: This file includes prediction performance scores for each fold of the 5-fold cross validation.
    - Drug-target protein family classification:
      - drug_target_family_pred_f1_perclass_representation_name.npy: This file includes F1-scores for each of the 100 runs of the repeated model train/test, which uses different randomly generated dataset splits in the class-based prediction setting, for the selected representation method.
      - drug_target_family_pred_accuracy_perclass_representation_name.npy: This file includes Accuracy scores for each of the 100 runs of the repeated model train/test, which uses different randomly generated dataset splits in the class-based prediction setting, for the selected representation method.
      - drug_target_family_pred_mcc_perclass_representation_name.npy: This file includes MCC scores for each of the 100 runs of the repeated model train/test, which uses different randomly generated dataset splits in the class-based prediction setting, for the selected representation method.
  
- **Benchmarking your own representation model:**

  - Semantic similarity prediction, Ontology-based protein function prediction and Drug target protein family classification tasks can be run for any representation vector dataset. There are two possible ways to do this:
    - Cloning the capsule and running it on Code Ocean. 
    - Clonning this repository  (this option is advised if you plan to run additional tasks over the default ones, as the run time may be elevated).
  
  - Prepraration of the input vector dataset: 
    - Generate your representation vectors for all human proteins (i.e. canonical isoforms).
    - File format:
      - Each row corresponds to the representation vector of a particular protein.
      - The first column header should be "Entry" and the rows contain the respective UniProt protein accessions.
      - Following column headers can be the index number of the dimension of the vector, the rows contain representation vectors' values (i.e. each column corresponds to a dimension of the representation vector).
      - Representation vectors should have a fixed size.
      - Save your representation vectors in comma separated (csv) text file.
      - Example files can be found in the folder: ../data/representation_vectors.
  - Following the generation of the representation vector file, the benchmark tests can be run as described above.
  - For the local run, dependencies can be found in the environment section.


# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
