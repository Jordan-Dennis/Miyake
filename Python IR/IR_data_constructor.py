from os import getcwd, walk
from pandas import read_csv, concat

data_sets = []  # List for storing the data set locations
data_sets_directory = f"{getcwd()}/datasets" # Home directory of the data 
for (root, dirs, files) in walk(data_sets_directory):    # Looping over directories 
    for file in files:  # Looping through the files 
        file_path = root + "/" + file   # Setting up the path 
        data_sets.append(file_path)  # Extending the stored directoriess

mixed_res = read_csv("Intcal20.csv")    # Loading the Intcal20 data
mixed_res = mixed_res.assign(id="Intcal20") # Assigning it an Id

pd_concat_iter = [] # Iterable holding DataFrames corresponding to single year datasets 

for data_set in data_sets:
    iter_set = read_csv(data_set)
    iter_set_id = data_set.split("/")[-1].strip(".csv") # id in format AuthorYY_Species
    iter_set = iter_set.assign(id=iter_set_id)
    pd_concat_iter.append(iter_set)

annual_res = concat(pd_concat_iter) # Constructing the dataset

def correct_sign(row):
    """
    Corrects the sign of the Brehm data which was recorded without the minus sign for BCE
    """
    if row["id"][:7] == "Brehm21":
        row["year"] = -row["year"]
    return row

annual_res = annual_res.apply(correct_sign, axis=1) # Fixing the signs

concat([mixed_res, annual_res]).to_csv("IR_Data.csv")