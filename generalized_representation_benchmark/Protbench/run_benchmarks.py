import argparse
import SimilarityCorrelation as smc
import GOPredictor as gp
import pandas as pd
import tqdm

parser = argparse.ArgumentParser(description='A Protein Representation Benchmark Suite')
parser.add_argument("-s","--similarity", action='store_true', help="Protein similarity based correlation benchmark")
parser.add_argument("-c", "--classification", action='store_true',  help="Protein classification based benchmark")
parser.add_argument("-f", "--function", action='store_true', help="Protein function prediction based benchmark")
parser.add_argument("-a", "--all", action='store_true',  help="Run all benchmarks")
parser.add_argument("-sts","--similarity_tasks", choices= ["Sparse","200","500","All"], default='Sparse')
parser.add_argument("-fpa","--function_prediction_aspect", choices= ["MF","BP","CC","All_Aspects"], default='BP')
parser.add_argument("-fpd","--function_prediction_dataset", choices= ["High","Middle","Low","All_Data_Sets"], default='Low')
parser.add_argument("-rf", "--representation_file", required=True,  help="File name of the representation - It should be a pandas dataframe in pickle format which includes 'Entry' and 'Vector' columns ")
parser.add_argument("-rn", "--representation_name", required=True, help="Name of the representation will shown on charts and graphics")

try:
    args = parser.parse_args()
    if not (args.similarity or args.classification or args.function or args.all):
            parser.error('At least one benchmark type should be selected')
except:
    parser.print_help()

print(args)

def load_representation(multi_col_representation_vector_file_path):
    print("Representation Vector is Loading...")
    multi_col_representation_vector = pd.read_csv(multi_col_representation_vector_file_path)
    vals = multi_col_representation_vector.iloc[:,1:(len(multi_col_representation_vector.columns))]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_vector.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

representation_dataframe = load_representation(args.representation_file)
 
if args.similarity or args.all:
    print("Protein Similarity Calculation Started...")
    smc.representation_dataframe = representation_dataframe
    smc.representation_name = args.representation_name
    smc.representation_dataframe = load_representation(args.representation_file)
    smc.representation_name = args.representation_name
    # The representation should be a pickle object which is a dataframe consists of two coloumns
    # Entry (UNIPROT Entry_ID) and Vector (the representation vector belongs to that protein)
    smc.protein_names = smc.representation_dataframe['Entry'].tolist()
    smc.similarity_tasks = args.similarity_tasks
    smc.calculate_all_correlations()
if args.function or args.all:
    print("Protein Function Prediction Started...")
    gp.aspect_type = args.function_prediction_aspect
    gp.dataset_type = args.function_prediction_dataset
    gp.representation_dataframe = representation_dataframe
    gp.representation_name = args.representation_name
    gp.pred_output()



