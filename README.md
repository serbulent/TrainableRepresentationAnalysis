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

- **go_prediction** folder is under **Benchmark_Study** and includes codes used for "Ontology-based protein function prediction" task.

- **embedding_similarity** folder is under **Benchmark_Study** and includes codes used for "Semantic similarity inference" task.

- **protein_family_prediction** folder is under **Benchmark_Study** and includes codes used for "Drug-target protein family classification" task.

- **protein_affinity_prediction** folder is under **Benchmark_Study** and includes codes used for "Protein-protein binding affinity estimation" task.

- **data/preprocess** folder is under **Benchmark_Study** and includes codes used for data preprocessing for "Ontology-based protein function prediction" and "Semantic similarity inference" tasks.

- The large files used for the benchmark is shared on [GDrive](https://drive.google.com/drive/folders/1adgnOlb-4gQLgxEdsFmwtYFoYaiq5Eva) and the main directory structure is shown below.

The data needed by the PROBE tool is located under "PROBE" folder. The data belongs to benchmark study was located under the "Benchmark_IO_data" folder. Each task and  shared data has its own folder. The directory structure is self-explanatory and standard for all tasks, hence some of the folders might be empty based on task for now. 

Other than that the reusable 23 precalculated reusable protein representation vectors for human proteins can be found under **"Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/HUMAN_PROTEIN_VECTORES"** folder for all tasks except protein-protein binding affinity estimation. The respresentations of proteins belongs to protein-protein binding affinity estimation task is in **"Benchmark_IO_Data/Shared_Data/Representations/representations_vectors/representation_vector_dataframes/SKEMPI"** folder for further use.

<pre>
-PROBE 
-Benchmark_IO_data
|
|---GO_prediction
   |
   |---preprocess
   |---input_data
   |---results
      |---main_results
      |---detailed_results
|---Embedding_Similarity
   |
   |---preprocess
   |---input_data
   |---results
      |---main_results
      |---detailed_results
|---Protein_Family_Prediction
   |
   |---preprocess
   |---input_data
   |---results
      |---main_results
      |---detailed_results
|---Shared_Data
</pre>

# License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
