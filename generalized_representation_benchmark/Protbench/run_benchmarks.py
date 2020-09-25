import argparse
import SimilarityCorrelation as smc
import pandas as pd

parser = argparse.ArgumentParser(description='A Protein Representation Benchmark Suite')
parser.add_argument("-s","--similarity", action='store_true', help="Protein similarity based correlation benchmark")
parser.add_argument("-c", "--classification", action='store_true',  help="Protein classification based benchmark")
parser.add_argument("-f", "--function", action='store_true', help="Protein function prediction based benchmark")
parser.add_argument("-a", "--all", action='store_true',  help="Run all benchmarks")
parser.add_argument("-rf", "--representation_file", required=True,  help="File name of the representation - It should be a pandas dataframe in pickle format which includes 'Entry' and 'Vector' columns ")
parser.add_argument("-rn", "--representation_name", required=True, help="Name of the representation will shown on charts and graphics")

try:
    args = parser.parse_args()
    if not (args.similarity or args.classification or args.function or args.all):
            parser.error('At least one benchmark type should be selected')
except:
    parser.print_help()

print(args)

if args.similarity or args.all:
    smc.representation_dataframe = pd.read_pickle(args.representation_file)
    smc.representation_name = args.representation_name
    # The representation should be a pickle object which is a dataframe consists of two coloumns
    # Entry (UNIPROT Entry_ID) and Vector (the representation vector belongs to that protein)
    smc.protein_names = smc.representation_dataframe['Entry'].tolist()
    smc.calculate_all_correlations()



