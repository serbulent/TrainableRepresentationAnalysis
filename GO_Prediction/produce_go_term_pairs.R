library(GO.db)
library(stringr)

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

produce_go_term_pairs <- function(ontology_type){
	# Lin measure is choosed since Lin and Resnik gives best results and 
	# others have bugs which produces results over 1 in GOSemSim library
       	# see https://arxiv.org/pdf/1310.8059.pdf for benchmarks.
	if(ontology_type == "MF" )
		term_list = all_MF_terms
	if(ontology_type == "BP" )
		term_list = all_BP_terms
	if(ontology_type == "CC" )
		term_list = all_CC_terms

	go_term_pairs = expand.grid(term1=term_list,term2=term_list)
	output_file_tmp = cbind(ontology_type,"GO_Term_Pairs.tsv")
	output_file = str_c(output_file_tmp, collapse="_")
	write.table(go_term_pairs, file =output_file,row.names=FALSE, sep = " ",quote = FALSE,col.names=FALSE)
}

setwd("/media/DATA/serbulent/Code/Thesis/ReviewPaper/GO_Prediction")
produce_go_term_pairs("MF")
produce_go_term_pairs("CC")
produce_go_term_pairs("BP")


