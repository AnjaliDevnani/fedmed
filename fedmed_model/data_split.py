# data_split.py
"""
Split dataset across hospitals to simulate 
federated learning with real-world data heterogeneity.

Supports two split strategies:
  1. IID (Independent & Identically Distributed):
     Random split — each hospital gets similar class distribution.
     Easier for FL convergence. Used for initial testing.

  2. Non-IID (Realistic):
     Each hospital has different class distributions.
     Hospital A: mostly pneumonia cases (city hospital)
     Hospital B: more normal cases (screening center)
     Hospital C: balanced but fewer samples (research)
     This is more realistic and harder for FL.
"""
import pandas as pd
import numpy as np
import os
from config import cfg

def create_iid_split(df: pd.DataFrame, n_hospitals: int = 3, 
                     seed: int = 42) -> dict:
    """
    IID split: randomly divide data among hospitals.
    Each hospital gets similar class distribution.
    """
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    splits = np.array_split(df, n_hospitals)
    hospital_ids = ["A", "B", "C"][:n_hospitals]
    return dict(zip(hospital_ids, splits))


def create_non_iid_split(df: pd.DataFrame, seed: int = 42) -> dict:
    """
    Non-IID split: realistic data heterogeneity.
    
    Hospital A (City General):   40% of data, 70% Pneumonia cases
    Hospital B (St. Mary's):     35% of data, 30% Pneumonia cases  
    Hospital C (Apollo Research):25% of data, 50% Pneumonia cases
    """
    np.random.seed(seed)
    
    normal_df = df[df['binary_label'] == 0].sample(
        frac=1, random_state=seed
    )
    pneum_df  = df[df['binary_label'] == 1].sample(
        frac=1, random_state=seed
    )
    
    n_normal = len(normal_df)
    n_pneum  = len(pneum_df)
    
    # Hospital A: 40% total, heavy pneumonia
    ha_pneum  = pneum_df.iloc[:int(0.55 * n_pneum)]
    ha_normal = normal_df.iloc[:int(0.25 * n_normal)]
    hospital_A = pd.concat([ha_pneum, ha_normal]).sample(frac=1)
    
    # Hospital B: 35% total, mostly normal
    hb_pneum  = pneum_df.iloc[int(0.55*n_pneum):int(0.75*n_pneum)]
    hb_normal = normal_df.iloc[int(0.25*n_normal):int(0.60*n_normal)]
    hospital_B = pd.concat([hb_pneum, hb_normal]).sample(frac=1)
    
    # Hospital C: 25% total, balanced
    hc_pneum  = pneum_df.iloc[int(0.75 * n_pneum):]
    hc_normal = normal_df.iloc[int(0.60 * n_normal):]
    hospital_C = pd.concat([hc_pneum, hc_normal]).sample(frac=1)
    
    splits = {"A": hospital_A, "B": hospital_B, "C": hospital_C}
    
    # Print statistics
    for hid, hdf in splits.items():
        n = len(hdf)
        p = sum(hdf['binary_label'] == 1)
        print(f"Hospital {hid}: {n} samples | "
              f"Normal: {n-p} ({100*(n-p)/n:.1f}%) | "
              f"Pneumonia: {p} ({100*p/n:.1f}%)")
    
    return splits


def split_and_save(
    data_csv: str,
    output_dir: str,
    strategy: str = "non_iid",
    val_ratio: float = 0.2
):
    """
    Load dataset CSV, split across hospitals, save as CSV files.
    Each hospital gets: hospital_X_train.csv and hospital_X_val.csv
    """
    print(f"[DataSplit] Loading {data_csv}...")
    df = pd.read_csv(data_csv)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if strategy == "iid":
        splits = create_iid_split(df)
    else:
        splits = create_non_iid_split(df)
    
    for hospital_id, hospital_df in splits.items():
        # Train/val split per hospital
        n = len(hospital_df)
        n_val = int(n * val_ratio)
        hospital_df = hospital_df.sample(frac=1, random_state=42)
        
        val_df   = hospital_df.iloc[:n_val]
        train_df = hospital_df.iloc[n_val:]
        
        # Save
        train_path = f"{output_dir}/hospital_{hospital_id}_train.csv"
        val_path   = f"{output_dir}/hospital_{hospital_id}_val.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path,   index=False)
        
        print(f"Hospital {hospital_id}: "
              f"Train={len(train_df)} | Val={len(val_df)} "
              f"→ {output_dir}/")


if __name__ == "__main__":
    split_and_save(
        data_csv=f"{cfg.data_dir}/all_data.csv",
        output_dir=cfg.split_dir,
        strategy="non_iid"
    )
