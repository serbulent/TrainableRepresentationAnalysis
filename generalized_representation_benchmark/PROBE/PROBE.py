import yaml
import pandas as pd
import tqdm
import SimilarityCorrelation as smc
import Drug_target_class_prediction as dtcp
import GOPredictor as gp
import AffinityPredictor as afp

with open('probe_config.yaml') as f:
	args = yaml.load(f, Loader=yaml.FullLoader)

if args["benchmark"] not in ["similarity","family","function","affinity","all"]:
        parser.error('At least one benchmark type should be selected')

print(args)

def load_representation(multi_col_representation_vector_file_path):
    multi_col_representation_vector = pd.read_csv(multi_col_representation_vector_file_path)
    vals = multi_col_representation_vector.iloc[:,1:(len(multi_col_representation_vector.columns))]
    original_values_as_df = pd.DataFrame({'Entry': pd.Series([], dtype='str'),'Vector': pd.Series([], dtype='object')})
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_vector.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

if args["benchmark"] in  ["similarity","function","all"]:
    print("Representation Vector is Loading... \n\n")
    representation_dataframe = load_representation(args["representation_file_human"])
 
if args["benchmark"] in  ["similarity","all"]:
    print("\n\nProtein Similarity Calculation Started...\n")
    smc.representation_dataframe = representation_dataframe
    smc.representation_name = args["representation_name"]
    smc.protein_names = smc.representation_dataframe['Entry'].tolist()
    smc.similarity_tasks = args["similarity_tasks"]
    smc.detailed_output = args["detailed_output"]
    smc.calculate_all_correlations()
if args["benchmark"] in  ["function","all"]:
    print("\n\n Ontology Based Protein Function Prediction Started...\n")
    gp.aspect_type = args["function_prediction_aspect"]
    gp.dataset_type = args["function_prediction_dataset"]
    gp.representation_dataframe = representation_dataframe
    gp.representation_name = args["representation_name"]
    gp.detailed_output = args["detailed_output"]
    gp.pred_output()
if args["benchmark"] in  ["family","all"]:
    print("\n\nDrug Target Protein Family Prediction Started...\n")
    dtcp.representation_path = args["representation_file_human"]
    dtcp.representation_name = args["representation_name"]
    dtcp.detailed_output = args["detailed_output"]
    dtcp.dataset = args["family_prediction_dataset"]
    dtcp.score_protein_rep()
if args["benchmark"] in  ["affinity","all"]:
    print("\n\nProtein Affinity Prediction Started...\n")
    afp.skempi_vectors_path = args["representation_file_affinity"]
    afp.representation_name = args["representation_name"]
    afp.predict_affinities_and_report_results()



