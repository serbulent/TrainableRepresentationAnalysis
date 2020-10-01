import pandas as pd
 
file_path = "uniprot_human_all.tab"
uniprot_vars = ['Entry','Entry name','Status','Protein names','Gene names','Organism','Length','Annotation' ]
uniprot_df = pd.read_csv(file_path, sep='\t', header=None, names=uniprot_vars)
#prot_ids = uniprot_df[uniprot_df['Annotation'] == '5 out of 5']
prot_ids = uniprot_df
print(prot_ids['Entry'])
prot_ids['Entry'].to_csv('human_all_well_annotated_proteins_accession_ids.tsv', sep='\t', index=False)

prot_ids['Entry name'].to_csv('human_all_well_annotated_proteins_entry_names.tsv', sep='\t')

