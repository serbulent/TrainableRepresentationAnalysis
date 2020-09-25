
#library(GOSemSim)
#hsGO <- godata('org.Hs.eg.db', ont=ontology_type, keytype = "UNIPROT")
#annotations_orgDb <- AnnotationDbi::select(org.Hs.eg.db, keys = res_tableOE_tb$gene, columns = c("SYMBOL", "ENTREZID","GENENAME"), keytype = "ENSEMBL")


x <- org.Hs.egGO
# Get the entrez gene identifiers that are mapped to a GO ID
mapped_genes <- mappedkeys(x)

xx <- as.list(x[mapped_genes])
if(length(xx) > 0) {
	# Try the first one
	got <- xx[[1]]
	org.Hs.egMAP
	got[[1]][["GOID"]]
	got[[1]][["Ontology"]]
	got[[1]][["Evidence"]]
}


hs_keys <- keys(org.Hs.eg.db,keytype="UNIPROT") 
anno_GO <- AnnotationDbi::select(org.Hs.eg.db, keytype = "UNIPROT", keys = hs_keys, columns = "GO")

print("Filtered GO Terms")
length(unique(subset(anno_GO, EVIDENCE!='IEA')[['GO']])) 
print("Filtered GO MF")
length(unique(subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='MF' )[['GO']])) 
print("Filtered GO BP")
length(unique(subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='BP' )[['GO']])) 
print("Filtered GO CC")
length(unique(subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='CC' )[['GO']])) 



print("Filtered Annotation Number")
subset_annot <- subset(anno_GO, EVIDENCE!='IEA')
nrow(unique(subset_annot[c('UNIPROT','GO')]))
print("Filtered Annotation Number MF")
subset_annot_MF <- subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='MF')
nrow(unique(subset_annot_MF[c('UNIPROT','GO')]))
print("Filtered Annotation Number BP")
subset_annot_BP <- subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='BP')
nrow(unique(subset_annot_BP[c('UNIPROT','GO')]))
print("Filtered Annotation Number CC")
subset_annot_CC <- subset(anno_GO, EVIDENCE!='IEA'& ONTOLOGY=='CC')
nrow(unique(subset_annot_CC[c('UNIPROT','GO')]))



head(anno_GO)
