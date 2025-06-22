import pandas as pd
import pickle
import glob
import os
import sys
import argparse
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="SimData", type=str, help="Directory location for data to process")
parser.add_argument("--output_dir", default="processed_split_sequences/", type=str, help="Location to store the output processed data")


def process_config_data(config_lines: list=None):
    """
    This function takes in a list of strings from the SimConfig.txt files for each simulation run.
    Each line should be read into a list, and that list gets passed into here.
    
    IF NEW METADATA FEATURES ARE ADDED OR REMOVED, ADJUST THIS FUNCTION ACCORDINGLY!
    
    Args:
        config_lines: a list of strings, each line of the text file being an individual item in the list.
    """
    config_lines = [line.split(":")[-1].replace(" ", "").replace("\n", "") for line in config_lines]
    
    # Dimer active state conversion to float
    if config_lines[26] == "False":
        config_lines[26] = 0.0
    else:
        config_lines[26] = 1.0
    
    # GTP/GDP active state conversion to float
    if config_lines[35] == "False":
        config_lines[35] = 0.0
    else:
        config_lines[35] = 1.0
    
    # Katanin active state conversion to float
    if config_lines[44] == "False":
        config_lines[44] = 0.0
    else:
        config_lines[44] = 1.0
    config_data = {
        "brownian_force": float(config_lines[2]),
        "dimer_count": float(config_lines[17]),
        "dimer_max_dehydro_delay": float(config_lines[18]),
        "dimer_min_dehydro_delay": float(config_lines[19]),
        "max_break_iterations": float(config_lines[20]),
        "dimer_bond_affinity_range": float(config_lines[21]),
        "break_prob_1": float(config_lines[22]),
        "break_prob_2": float(config_lines[23]),
        "break_prob_3": float(config_lines[24]),
        "break_prob_4": float(config_lines[25]),
        "dimer_active_state": config_lines[26],
        "dimer_intro_iterations": float(config_lines[27]),
        "gtp_count": float(config_lines[32]),
        "gtp_max_dehydro_delay": float(config_lines[33]),
        "gtp_min_dehydro_delay": float(config_lines[34]),
        "gtp_active_state": config_lines[35],
        "gtp_intro_iterations": float(config_lines[36]),
        "katanin_count": float(config_lines[41]),
        "katanin_max_dehydro_delay": float(config_lines[42]),
        "katanin_min_dehydro_delay": float(config_lines[43]),
        "katanin_active_state": config_lines[44],
        "katanin_intro_iterations": float(config_lines[45]),
        "katanin_bond_affinity_range": float(config_lines[46])
    }
    return config_data


def scale_metadata(directory_list: list, train: bool=None):
    """
    This function searches each subdirectory in the directory_list to find the SimConfig.txt file.
    Then it uses the process_config_data function to gather the relevant data points into a dictionary.
    That dictionary is thrown into a list. This is done for every SimConfig.txt file in all subdirectories.
    
    The resulting list of dictionaries goes into a dataframe. All of the non-binary columns are scaled
    using a MinMaxScaler. If train = True, then the scaler is fit to this data and then transformed. If train
    = False, then the data is just transformed. This means train = True has to happen first in order to read in
    the right scaler to transform the test data.
    
    Once the data is scaled accordingly, each row (corresponding to 1 scaled simulator configuration) is written
    back to the same directory in a new .txt file called 'scaled_metadata.txt'.
    
    Args:
        directory_list (list): a list of paths to subdirectories. Each of these subdirectories should contain
            some number of simulation data in the form of 1 or more .csv files. There should also be 1 SimConfig.txt file 
            in these subdirs.
        train (bool): If set to True, the scaler will be fit to this data before transforming. Otherwise, data
            is only transformed.
    """
    config_dicts = []
    
    print("Finding current metadata")
    for sub_dir in directory_list:
        for file in os.listdir(sub_dir):
            if file == "SimConfig.txt":
                txt_file = os.path.join(sub_dir, file)
                
                with open(txt_file, 'r') as f:
                    content = f.readlines()
                config_dict = process_config_data(content)
                config_dict["file_name"] = txt_file
                config_dicts.append(config_dict)
    
    config_df = pd.DataFrame(config_dicts)
    no_scale_cols = [
        'break_prob_1',
        'break_prob_2',
        'break_prob_3',
        'break_prob_4',
        'dimer_active_state',
        'gtp_active_state',
        'katanin_active_state',
        'file_name'
    ]
    scale_cols = [col for col in config_df.columns if col not in no_scale_cols]
    
    if train:
        print("Fit-transforming metadata")
        scaler = MinMaxScaler()
        config_df_scaled = config_df.copy()
        config_df_scaled[scale_cols] = scaler.fit_transform(config_df[scale_cols])
        joblib.dump(scaler, 'minmax_scaler.pkl')
    else:
        print("Just transforming metadata")
        scaler = joblib.load('minmax_scaler.pkl')
        config_df_scaled = config_df.copy()
        config_df_scaled[scale_cols] = scaler.transform(config_df[scale_cols])
    
    print("Writing new metadata")
    for i, row in config_df_scaled.iterrows():
        file_name = row['file_name']
        sub_dir_path = file_name.split('/')[:-1]
        sub_dir_path = "/".join(sub_dir_path)
        new_file = f"{sub_dir_path}/scaled_metadata.txt"
        print(f"New file name: {new_file}")
        with open(new_file, 'w') as f:
            for col, val in row.items():
                if col != "file_name":
                    f.write(f"{val}\n")


def make_sequences(directory_list: list, train: bool=None):
    """
    This funciton takes in a list of intermediate directories and creates
    MT sequences out of the files within these directories. In each of the 
    directories of 'directory_list' there should be 1 .txt file for the sim
    config data and a few (typically 3) .csv files containing the simulation runs.
    """
    print(f"Making train={train} sequences")
    print("Scaling metadata")
    scale_metadata(directory_list, train=train)
    
    all_sequences = []
    for sub_dir in directory_list:
        txt_file = None
        csv_files = []
        for file in os.listdir(sub_dir):
            if file.endswith("scaled_metadata.txt"):
                txt_file = os.path.join(sub_dir, file)
            elif file.endswith(".csv"):
                csv_files.append(os.path.join(sub_dir, file))
            
        if not txt_file or not csv_files:
            print(f"Skipping {sub_dir} (missing .txt or .csv files)")
            break

        with open(txt_file, "r", encoding="utf-8") as f:
            txt_data = f.readlines()
            config_items = [float(item.strip()) for item in txt_data]

        for j, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            df = df.dropna().reset_index(drop=True)
            sequence = [config_items]
            for _, row in df.iterrows():
                sequence.append(row[' mtRepresentation'])
            all_sequences.append(sequence)
    return all_sequences


def main(args):
    # Make the output directory
    if not os.path.exists(args.output_dir):
        print("Making new output directory!")
        os.makedirs(args.output_dir)
    
    base_dir = args.data_dir
    
    intermediate_dirs = []
    for item in os.listdir(base_dir):
        if item == ".ipynb_checkpoints":
            continue
        item_path = os.path.join(base_dir, item)
        intermediate_dirs.append(item_path)
        
    print("Splitting subdirectories up for train-test split... ")
    train_dirs, test_dirs = train_test_split(intermediate_dirs, test_size=0.2, random_state=42)
    
    train_sequences = make_sequences(train_dirs, train=True)
    test_sequences = make_sequences(test_dirs, train=False)
    
    with open(args.output_dir + f"train_sequences_1.pkl", "wb") as f:
        pickle.dump(train_sequences, f)
        
    with open(args.output_dir + f"test_sequences_1.pkl", "wb") as f:
        pickle.dump(test_sequences, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)