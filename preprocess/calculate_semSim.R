#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install("GOSemSim")
#BiocManager::install("org.Hs.eg.db")
library(GOSemSim)
library(stringr)
library(parallel)
library(MASS)

calculate_similarities <- function(ontology_type){
	hsGO <- godata('org.Hs.eg.db', ont=ontology_type, keytype = "UNIPROT")
	# org.Hs.eg.db version 3.8.2 and AnnotationDbi version 1.46.1 was used for GO annotations
	input_file_tmp= cbind('uniprot_accession_ids_for_highestAnnotatedProteins_200_',ontology_type,'.tsv')
       	input_file = str_c(input_file_tmp, collapse="")
	proteins = read.table(file = input_file , sep = ' ', header = FALSE)
	# Rel measure is choosed since Lin and Resnik gives best results and 
	#Rel is combination of two of them see https://arxiv.org/pdf/1310.8059.pdf for benchmarks.
	proteinSimilarityMatrix = mgeneSim(genes=proteins[,1] , semData=hsGO, measure="Lin",verbose=TRUE, drop = "IEA" )
	output_file_tmp = cbind("human",ontology_type,"proteinSimilarityMatrix_for_highest_annotated_200_proteins.csv")
	output_file = str_c(output_file_tmp, collapse="_")
	write.csv(proteinSimilarityMatrix, file =output_file,row.names=FALSE)
}

setwd("/media/DATA/serbulent/Code/Thesis/ReviewPaper/preprocess")
#calculate_similarities("MF")
calculate_similarities("CC")
#calculate_similarities("BP")

#mclapply(c("MF","CC","BP"), calculate_similarities, mc.cores = 3)

