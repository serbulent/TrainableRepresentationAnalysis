library(org.Hs.eg.db)

fileName = "/media/DATA/serbulent/Code/Thesis/ReviewPaper/preprocess/GOTerms_4_test.csv" 
GOTerms4TestTable = read.csv2(fileName, header = TRUE, sep = ",",comment.char = '"',skip=2)
goterms = GOTerms4Test[,3]
goterms[1]

gene_list <- data.frame(mget(as.character(goterms[1]), org.Hs.egGO2ALLEGS)[[1]])
gene <- list <- data.frame(mget("GO:0072599", org.Hs.egGO2ALLEGS)[[1]])
gene_list <- data.frame(mget("GO:0015278", org.Hs.egGO2ALLEGS)[[1]]) 
gene_list
