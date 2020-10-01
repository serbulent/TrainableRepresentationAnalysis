#if (!requireNamespace("BiocManager", quietly = TRUE))
#	    install.packages("BiocManager")
#BiocManager::install("GOSim")

library(GO.db)
library(GOSim)
library(stringr)
library(parallel)
library(MASS)

## select() interface:
## Objects in this package can be accessed using the select() interface
## from the AnnotationDbi package. See ?select for details.
## Bimap interface:
# Convert the object to a list
all_BP_terms <- names(as.list(GOBPCHILDREN))
length(all_BP_terms)
all_MF_terms <- names(as.list(GOMFCHILDREN))
length(all_MF_terms)
all_CC_terms <- names(as.list(GOCCCHILDREN))
length(all_CC_terms)

all_CC_terms[0:100]

calculate_similarities <- function(ontology_type){
	# Lin measure is choosed since Lin and Resnik gives best results and 
	# others have bugs which produces results over 1 in GOSemSim library
       	# see https://arxiv.org/pdf/1310.8059.pdf for benchmarks.
	if(ontology_type == "MF" )
		term_list = all_MF_terms
	if(ontology_type == "BP" )
		term_list = all_BP_terms
	if(ontology_type == "CC" )
		term_list = all_CC_terms[0:100]

	proteinSimilarityMatrix = getTermSim(term_list, method="GIC", verbose=FALSE )
	output_file_tmp = cbind(ontology_type,"GO_Term_SimilarityMatrix.csv")
	output_file = str_c(output_file_tmp, collapse="_")
	write.csv(proteinSimilarityMatrix, file =output_file,row.names=FALSE)
}

setwd("/media/DATA/serbulent/Code/Thesis/ReviewPaper/GO_Prediction")
#calculate_similarities("MF",all_MF_terms)
calculate_similarities("CC")
#calculate_similarities("BP",all_BP_terms)

#mclapply(c("MF","CC","BP"), calculate_similarities, mc.cores = 3)

getTermSim(c("GO:0000109","GO:0000110"), method="GIC", verbose=FALSE )

