import argparse
import SimilarityCorrelation as smc
import Drug_target_class_prediction_general as dtcpg
import Drug_target_class_prediction as dtcp
import GOPredictor as gp
import AffinityPredictor as afp
import pandas as pd
import tqdm

parser = argparse.ArgumentParser(description='A Protein Representation Benchmark Suite')
parser.add_argument("-s","--similarity", action='store_true', help="Protein similarity based correlation benchmark")
parser.add_argument("-fam", "--family_prediction", action='store_true',  help="Protein family prediction benchmark")
parser.add_argument("-f", "--function_prediction", action='store_true', help="Protein function prediction benchmark")
parser.add_argument("-aff", "--affinity_prediction", action='store_true', help="Protein affinity prediction benchmark")
parser.add_argument("-a", "--all", action='store_true',  help="Run all benchmarks")
parser.add_argument("-d","--detailed_output", action='store_true', help="Save detailed outputs for tasks")
parser.add_argument("-sts","--similarity_tasks", choices= ["Sparse","200","500","All"], default='Sparse')
parser.add_argument("-fpa","--function_prediction_aspect", choices= ["MF","BP","CC","All_Aspects"], default='All_Aspects')
parser.add_argument("-fpd","--function_prediction_dataset", choices= ["High","Middle","Low","All_Data_Sets"], default='Low')
parser.add_argument("-rf", "--representation_file", required=True,  help="File name of the representation - It should be a pandas dataframe in pickle format which includes 'Entry' and 'Vector' columns ")
parser.add_argument("-rn", "--representation_name", required=True, help="Name of the representation will shown on charts and graphics")

try:
    args = parser.parse_args()
    if not (args.similarity or args.family_prediction or args.function_prediction or args.affinity_prediction or args.all):
            parser.error('At least one benchmark type should be selected')
except:
    parser.print_help()

print(args)
def load_representation(multi_col_representation_vector_file_path):
    multi_col_representation_vector = pd.read_csv(multi_col_representation_vector_file_path)
    vals = multi_col_representation_vector.iloc[:,1:(len(multi_col_representation_vector.columns))]
    original_values_as_df = pd.DataFrame({'Entry': pd.Series([], dtype='str'),'Vector': pd.Series([], dtype='object')})
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_vector.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

if args.similarity or args.function_prediction or args.all:
    print("Representation Vector is Loading... \n\n")
    representation_dataframe = load_representation(args.representation_file)
 
if args.similarity or args.all:
    print("\n\nProtein Similarity Calculation Started...\n")
    smc.representation_dataframe = representation_dataframe
    smc.representation_name = args.representation_name
    smc.representation_dataframe = representation_dataframe
    smc.representation_name = args.representation_name
    # The representation should be a pickle object which is a dataframe consists of two coloumns
    # Entry (UNIPROT Entry_ID) and Vector (the representation vector belongs to that protein)
    smc.protein_names = smc.representation_dataframe['Entry'].tolist()
    smc.similarity_tasks = args.similarity_tasks
    smc.calculate_all_correlations()
if args.function_prediction or args.all:
    print("\n\n Ontology Based Protein Function Prediction Started...\n")
    gp.aspect_type = args.function_prediction_aspect
    gp.dataset_type = args.function_prediction_dataset
    gp.representation_dataframe = representation_dataframe
    gp.representation_name = args.representation_name
    gp.detailed_output = args.detailed_output
    gp.pred_output()
if args.family_prediction or args.all:
    print("\n\nDrug Target Protein Family Prediction Started...\n")
    dtcp.representation_path = args.representation_file
    dtcp.representation_name = args.representation_name
    dtcp.detailed_output = args.detailed_output
    dtcp.score_protein_rep()
    dtcpg.representation_path = args.representation_file
    dtcpg.representation_name = args.representation_name
    dtcpg.score_protein_rep()
if args.affinity_prediction or args.all:
    print("\n\nProtein Affinity Prediction Started...\n")
    afp.skempi_vectors_path = args.representation_file
    afp.representation_name = args.representation_name
    afp.predict_affinities_and_report_results()


