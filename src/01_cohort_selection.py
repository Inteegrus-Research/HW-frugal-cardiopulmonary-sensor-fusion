"""
01_cohort_selection.py
----------------------
Objective: Extract a reproducible, unstratified cohort from the MESA Sleep Dataset 
to preserve natural epidemiological class imbalance. Generates a bash script for 
targeted downloading via the NSRR API.
"""

import pandas as pd
import os
import argparse

def main():
    # File paths
    input_file = '../data/mesa-sleep-dataset-0.8.0.csv'
    output_cohort = '../data/cohort_list.csv'
    output_script = '../download_edfs.sh'
    target_size = 800
    random_seed = 42

    print(f"Loading demographic metadata from: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing master MESA csv at {input_file}. Please download from NSRR.")

    df = pd.read_csv(input_file)

    # Clean data: drop rows missing required clinical labels
    df_clean = df.dropna(subset=['ahi_a0h4', 'sleepage5c', 'bmi5c'])

    # Unstratified random sampling
    df_cohort = df_clean.sample(n=target_size, random_state=random_seed)

    # Save cohort list
    os.makedirs('../data', exist_ok=True)
    df_cohort.to_csv(output_cohort, index=False)
    print(f"Cohort demographics saved to: {output_cohort}")

    # Generate NSRR API download script
    with open(output_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated NSRR download script for N={target_size} cohort\n\n")
        
        for _, row in df_cohort.iterrows():
            patient_id = str(int(row['mesaid'])).zfill(4)
            f.write(f"nsrr download mesa/polysomnography/edfs/mesa-sleep-{patient_id}.edf -d data/raw_edfs/\n")

    print(f"NSRR download script generated at: {output_script}")
    print("Execute this script to download raw EDF files into data/raw_edfs/")

if __name__ == "__main__":
    main()
