## Organization of this repository

- PROBE (Protein RepresentatiOn BEnchmark) is tool that can be used to evaluate any protein representation model. The tool can be found under 
**[bin](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/bin)** folder. Also study results can be reproduced easily using PROBE.

- **[paper_reproduction_scripts](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts)** folder includes codes used for benchmarks presented in the paper.

- **[function_prediction](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/function_prediction)** folder is under **paper_reproduction_scripts** and includes codes used for "Ontology-based protein function prediction" task.

- **[semantic_similarity_inference](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/semantic_similarity_inference)** folder is under **paper_reproduction_scripts** and includes codes used for "Semantic similarity inference" task.

- **[target_family_classification](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/target_family_classification)** folder is under **paper_reproduction_scripts** and includes codes used for "Drug-target protein family classification" task.

- **[binding_affinity_estimation](https://github.com/serbulent/TrainableRepresentationAnalysis/tree/master/paper_reproduction_scripts/binding_affinity_estimation)** folder is under **paper_reproduction_scripts** and includes codes used for "Protein-protein binding affinity estimation" task.

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
