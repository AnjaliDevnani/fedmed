import os
import glob
import pandas as pd
from config import cfg
import random

def create_csvs_from_folder():
    """Reads Kaggle dataset folders and creates CSVs."""
    base_dir = "./data/chest_xray"
    
    # Check if download exists, if not generate dummy data
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Generating dummy dataset for testing...")
        import numpy as np
        from PIL import Image
        for split in ["train", "val", "test"]:
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = os.path.join(base_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                for i in range(10): # 10 images per class/split
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    Image.fromarray(img).save(os.path.join(class_dir, f"dummy_{i}.jpg"))

    data = []
    
    # Process train, val, test folders
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            label = "Normal" if class_name == "NORMAL" else "Pneumonia"
            binary_label = 0 if class_name == "NORMAL" else 1
            
            # Find all images
            for img_path in glob.glob(os.path.join(class_dir, "*.[jJ][pP]*[gG]")):
                data.append({
                    "image_path": os.path.abspath(img_path), 
                    "label": label, 
                    "binary_label": binary_label,
                    "split": split
                })

    df = pd.DataFrame(data)
    if len(df) == 0:
        print("No images found to preprocess.")
        return

    os.makedirs(cfg.data_dir, exist_ok=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save all_data.csv for data_split.py
    df.to_csv(os.path.join(cfg.data_dir, "all_data.csv"), index=False)
    print(f"Created all_data.csv with {len(df)} records.")
    
    # Also save train.csv and val.csv for central training
    # Many users use 80/20 train/val split if the original val is too small (kaggle val is just 16 images)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df.to_csv(os.path.join(cfg.data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.data_dir, "val.csv"), index=False)
    
    print(f"Created train.csv ({len(train_df)}) and val.csv ({len(val_df)}).")

if __name__ == "__main__":
    create_csvs_from_folder()
