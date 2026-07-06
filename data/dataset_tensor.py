#loading, padding and normalization
#saving the results as tensors 

import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Configuration
COMPATIBLE_SIZE = (128, 160, 128)
P99_9 = 1.0437
OUTPUT_DIR = "./databases/PREPROCESSED_TENSORS_128x160x128_pretraining" 

data_folders = [
    "./datasets/CamCan",
    "./datasets/DLBS",
    "./datasets/IXI",
    "./datasets/NKI",
    "./datasets/OASIS",
    "./datasets/SALD",
    "./datasets/Wayne"
]

def collect_nifti_paths(folder_list):
    all_files = []
    print(f"Searching for NIfTI files in {len(folder_list)} folders...")
    for folder in folder_list:
        path_obj = Path(folder)
        files_in_folder = list(path_obj.rglob("*.nii.gz")) + list(path_obj.rglob("*.nii"))
        dataset_name = path_obj.name 
        for f in files_in_folder:
            all_files.append({
                "original_path": str(f),
                "filename": f.name,
                "dataset_source": dataset_name
            })
    return all_files

def preprocess_and_save(row):
    original_path = row["original_path"]
    filename = row["filename"]
    dataset_source = row["dataset_source"]
    
    # unique name, no overwriting
    tensor_filename = f"{dataset_source}_{filename}.pt"
    output_path = os.path.join(OUTPUT_DIR, tensor_filename)
    
    # will not run if it already exists
    if os.path.exists(output_path):
        return output_path

    try:
        # Loading Niftii
        img = nib.load(original_path).get_fdata().astype(np.float32)
        if img.ndim == 4: 
            img = img[..., 0]
        
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Global normalization
        img_tensor = torch.clamp(img_tensor, min=0.0)
        img_tensor = img_tensor / P99_9
        img_tensor = torch.clamp(img_tensor, max=1.0)
        
        # padding
        d, h, w = img_tensor.shape[1:]
        pad_d = max(0, COMPATIBLE_SIZE[0] - d)
        pad_h = max(0, COMPATIBLE_SIZE[1] - h)
        pad_w = max(0, COMPATIBLE_SIZE[2] - w)
        
        padding = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        )
        img_tensor = F.pad(img_tensor, padding, "constant", 0)
        
        # cropping
        img_tensor = img_tensor[:, :COMPATIBLE_SIZE[0], :COMPATIBLE_SIZE[1], :COMPATIBLE_SIZE[2]]
        
        # saving as PyTorch tensor
        torch.save(img_tensor.clone(), output_path)
        return output_path
        
    except Exception as e:
        print(f"\nERROR Processing {original_path}: {e}")
        return None

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files_metadata = collect_nifti_paths(data_folders)
    df = pd.DataFrame(files_metadata)
    print(f"Total NIFTI found: {len(df)}")
    
    processed_paths = []
    
    # Progress bar :)
    print("Initializing preprocessing and saving tensors...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = preprocess_and_save(row)
        processed_paths.append(out_path)
        
    df["tensor_path"] = processed_paths
    
    # filtering errors
    df_clean = df.dropna(subset=["tensor_path"])
    
    # saving a csv file with the routes of the tensors
    csv_path = os.path.join(OUTPUT_DIR, "preprocessed_dataset.csv")
    df_clean.to_csv(csv_path, index=False)
    
    print(f"\n¡Preprocesamiento completado! Tensores guardados en: {OUTPUT_DIR}")
    print(f"Se ha generado un archivo de mapeo en: {csv_path}")
