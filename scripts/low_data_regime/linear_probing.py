
import os
import os
import getpass
import tempfile
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_predict, PredefinedSplit
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, roc_curve, auc
import umap
import re
import zipfile
import tarfile
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import json
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import joblib  # Para guardar los embeddings
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, balanced_accuracy_score, matthews_corrcoef
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.svm import LinearSVC
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, 
                             roc_auc_score, r2_score, mean_absolute_error, 
                             roc_curve, auc)
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
import sys

from resnet_databases_curriculum_ssim import ResNet3D_Autoencoder

from ViT3D_Autoencoder import ViT3D_Autoencoder_all_d

from architectures import ResNet_3D_SimCLR, BasicBlock3D, Bottleneck3D, ViT3D_SimCLR, ViT3D_MAE_Base_Inference, SimCLR_ViT3D_BrainIAC, ResNet18_3D_MAE, ViT3D_SimCLR_Base, PatchEmbed3D, PatchEmbed3D_BrainIAC, ViT3D_Autoencoder_age

from visualization_functions import extract_if_compressed, find_nii_files, get_subject_id, get_last_available_epoch, preprocess_image, mask_patches_fixed, save_comparison_slices, get_visualization_brain, save_reconstruction_plot 
from visualization_architectures import ResNet50_3D_UNet, ResNet34_3D_MAE, ResNet50_3D_MAE

def get_resnet18_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(BasicBlock3D, [2, 2, 2, 2], in_chans, proj_dim)

def get_resnet50_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(Bottleneck3D, [3, 4, 6, 3], in_chans, proj_dim)

def get_model(model_type, checkpoint_path, device):
    """Instancia el modelo completo (con decoder) y carga pesos."""
    
    # depending on the model
    if model_type == "ResNet18_3D_MAE":
        model = ResNet18_3D_MAE(in_chans=1)
        
    elif model_type == "ResNet34_3D_MAE":
        model = ResNet34_3D_MAE(in_chans=1)

    elif model_type == "ResNet50_3D_MAE":
        model = ResNet50_3D_MAE(in_chans=1)

    elif model_type == "UNETresnetL":
        model = ResNet50_3D_UNet(in_chans=1)

    elif model_type == "vit_mae_base":
        model = ViT3D_Autoencoder(
            img_size=(128, 160, 128),
            patch_size=16, 
            embed_dim=768, 
            depth=12,
            num_heads=12,
            decoder_embed_dim=512, 
            decoder_depth=8,
            decoder_num_heads=16
        )

    elif model_type == "vit_mae_small" or model_type == "vit_mae_3d":
        model = ViT3D_Autoencoder(
            img_size=(128, 160, 128),
            patch_size=16,
            embed_dim=512, 
            depth=6,
            num_heads=8
        )

    elif model_type == "vit_mae_all_d":
        model = ViT3D_Autoencoder_all_d(
            patch_size=16, 
            embed_dim=512, 
            depth=6
        )

    elif "simclr" in model_type:
        print(f" El modelo {model_type} es de tipo SimCLR y NO tiene decoder por defecto.")
       
        if "resnet18" in model_type:
            model = get_resnet18_3d(in_chans=1, proj_dim=128)
        else:
            model = ViT3D_SimCLR_Base(img_size=(128, 160, 128), patch_size=16)

    else:
        raise ValueError(f"Modelo {model_type} no soportado para visualización.")

    # loading checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if 'pos_embed' in state_dict and hasattr(model, 'pos_embed'):
        pos_embed_checkpoint = state_dict['pos_embed']
        pos_embed_model = model.pos_embed
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"Adapting pos_embed: {pos_embed_checkpoint.shape} -> {pos_embed_model.shape}")
            
            # token CLS management
            if pos_embed_checkpoint.shape[1] == pos_embed_model.shape[1] - 1:
                cls_token_pos = pos_embed_model[:, :1, :]
                pos_embed_checkpoint = torch.cat((cls_token_pos, pos_embed_checkpoint), dim=1)
            elif pos_embed_checkpoint.shape[1] == pos_embed_model.shape[1] + 1:
                pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
            
            # Spatial interpolation
            #if pos_embed_checkpoint.shape != pos_embed_model.shape:
                #state_dict['pos_embed'] = pos_embed_checkpoint # (después de interpolar)

    # Final loading
    model.load_state_dict(state_dict, strict=False) 
    # strict=False in case we use the weights from SimCLR 
    
    return model.to(device).eval()

def guardar_rids_splits(y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender, SEEDS):
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # original csv route
    csv_path = './final_metadata_interactions_v4.csv' 
    
    if not os.path.exists(csv_path):
        # FALLBACK: If the csv is not found, it will use the indices from rows (0, 1, 2...) that are mapped 1:1 with the Dataframe
        rids = np.arange(len(y_dx))
        print(f" CSV no encontrado en {csv_path}. Se usarán los índices de fila como IDs correlativos.")
    else:
        df_meta = pd.read_csv(csv_path)
        rids = df_meta['RID'].values

    output_dir = "rids_splits_cache"
    os.makedirs(output_dir, exist_ok=True)

    # Replicating the mask and label filters of each task
    mask_ad_cn = np.isin(y_dx, [0, 1, 2])
    y_ad_cn = np.where(np.isin(y_dx[mask_ad_cn], [0, 1]), 0, 1)

    mask_mci = y_mci != -1
    y_mci_task = y_mci[mask_mci]

    mask_amyloid = y_amyloid != -1
    y_amyloid_task = y_amyloid[mask_amyloid]

    mask_gender = y_gender != -1
    y_gender_task = y_gender[mask_gender]

    mask_m3 = np.isin(y_dx, [0, 1, 2, 4, 5])
    y_m3 = np.zeros(np.sum(mask_m3))
    y_m3[np.isin(y_dx[mask_m3], [0, 1])] = 0
    y_m3[np.isin(y_dx[mask_m3], [4, 5])] = 1
    y_m3[y_dx[mask_m3] == 2] = 2

    mask_phc = (y_phc != -1000.0) & (~np.isnan(y_phc))
    y_phc_task = y_phc[mask_phc]

    mask_age = (y_age > 0) & (~np.isnan(y_age))
    y_age_task = y_age[mask_age]

    tasks_config = {
        'AD_vs_CN': (mask_ad_cn, y_ad_cn, True),
        'sMCI_vs_pMCI': (mask_mci, y_mci_task, True),
        'Amyloid': (mask_amyloid, y_amyloid_task, True),
        'Gender': (mask_gender, y_gender_task, True),
        'Multi_3Class': (mask_m3, y_m3, True),
        'PHC_Reg': (mask_phc, y_phc_task, False),
        'Age_Reg': (mask_age, y_age_task, False)
    }

    fractions = [("1pc", 0.01), ("2pc", 0.02), ("8pc", 0.08), ("25pc", 0.25), ("50pc", 0.5), ("75pc", 0.75), ("100pc", 1.0)]

    print("\n [Alineación] Generando archivos CSV con la distribución exacta de RIDs...")

    for seed in SEEDS:
        for task_name, (mask, y_task, stratify) in tasks_config.items():
            rids_task = rids[mask]
            
            # SPLIT 1: Separating Test partition(25%)
            strat_1 = y_task if stratify else None
            rids_train_val, rids_test, y_train_val, _ = train_test_split(
                rids_task, y_task, test_size=0.25, random_state=seed, stratify=strat_1
            )
            
            # SPLIT 2: Separating Train and Val (1/3 of 75% = 25%)
            strat_2 = y_train_val if stratify else None
            rids_train, rids_val, y_train, _ = train_test_split(
                rids_train_val, y_train_val, test_size=1/3, random_state=seed, stratify=strat_2
            )
            
            # fraction loop (data scarcity)
            for porcentaje_str, f in fractions:
                if f < 1.0:
                    strat_f = y_train if stratify else None
                    rids_train_frac, _, _, _ = train_test_split(
                        rids_train, y_train, train_size=f, random_state=seed, stratify=strat_f
                    )
                else:
                    rids_train_frac = rids_train
                
                # Structuring the RIDs identifying what group they belong in
                df_train = pd.DataFrame({'RID': rids_train_frac, 'Set': 'Train'})
                df_val = pd.DataFrame({'RID': rids_val, 'Set': 'Validation'})
                df_test = pd.DataFrame({'RID': rids_test, 'Set': 'Test'})
                
                df_split_final = pd.concat([df_train, df_val, df_test], ignore_index=True)
                
                # saving individual csv
                filename = f"split_seed_{seed}_task_{task_name}_frac_{porcentaje_str}.csv"
                df_split_final.to_csv(os.path.join(output_dir, filename), index=False)

    print(f" Proceso completado. Todos los splits se han guardado en: '{output_dir}/'\n")

    
def evaluar_modelo_epoch(model_info, epoch, dataloader, embeddings_dir):
    """
    independent function
    """
    print(f"Iniciando evaluación: {model_info['name']} - Epoch {epoch}")
    
    # Extract or load embeddings
    X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender = get_or_extract_embeddings(
        epoch, 
        dataloader, 
        model_info["type"], 
        model_info["checkpoint_path"], 
        embeddings_dir
    )
    
    # run downstream evaluation
    results, preds, _, _, _, _, _, _, _ = evaluate_downstream(
        X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender
    )
  

    return {"modelo": model_info["name"], "epoch": epoch, "results": results}


def extract_features_from_model(model, model_type, x):
    """manages the different ourputs from each architecture"""
    if model_type == "resnet_autoencoder":
        _, z = model(x)
        if len(z.shape) == 2:
            return z
        elif len(z.shape) == 5:
            return z.mean(dim=(2,3,4))
        else:
            return z.view(z.size(0), -1) 
            
    elif model_type == "ResNet18_3D_MAE":
        _, z_global = model(x)
        return z_global
    
    elif model_type == "ResNet34_3D_MAE":
        _, z_global = model(x)
        return z_global

    elif model_type == "ResNet50_3D_MAE":
        _, z_global = model(x)
        return z_global

    elif model_type == "UNETresnetL":
        _, z_global = model(x)
        return z_global

    elif model_type == "simclr":
        h, _ = model(x)
        return h
    elif model_type == "vit_mae":
        _, _, z_global = model(x, mask_ratio=0.0)
        return z_global
    elif model_type == "resnet_simclr":
        h, _ = model(x)
        return h
    elif model_type == "vit_mae_all_d":
        _, _, z_global = model(x, mask_ratio=0.0, mask_size_voxels=16) 
        return z_global
    elif model_type == "vit_mae_base":
        z_global = model(x)
        return z_global
    elif model_type == "vit_mae_small":
        z_global = model(x)
        return z_global
    elif model_type == "vit_mae_3d":
        _, _, z_global = model(x, mask_ratio=0.0, mask_size_voxels=16)
        return z_global
    
    elif model_type == "simclr_base":
        h, _ = model(x)
        return h
    
    elif model_type in ["resnet18", "resnet50"]:
        h, _ = model(x)
        return h

    elif model_type == "vit_simclr_brainiac":
        h, _ = model(x)
        return h  


# extraction and embeddings
def get_or_extract_embeddings(epoch, dataloader, model_type, checkpoint_path, embeddings_dir):
    cache_file = f"{embeddings_dir}/features_epoch_{epoch}_v4_amyloid_mci_age_gender.pkl" 
    
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    
    print(f"Extrayendo características para epoch {epoch}...")
    model = get_model(model_type, checkpoint_path, DEVICE)
    
   
    latent_vectors, labels_dx_list, labels_phc_list, labels_amyloid_list, labels_mci_list, labels_age_list, labels_gender_list = [], [], [], [], [], [], []
    
    with torch.no_grad():
        
        for images, dxs, phcs, amyloids, mcis, ages, genders in tqdm(dataloader, leave=False):
            images = images.to(DEVICE)
            features = extract_features_from_model(model, model_type, images)
            
            latent_vectors.append(features.cpu().numpy())
            labels_dx_list.append(dxs.numpy())
            labels_phc_list.append(phcs.numpy())
            labels_amyloid_list.append(amyloids.numpy())
            labels_mci_list.append(mcis.numpy()) 
            labels_age_list.append(ages.numpy())
            labels_gender_list.append(genders.numpy())
            
    X = np.concatenate(latent_vectors, axis=0)
    y_dx = np.concatenate(labels_dx_list, axis=0)
    y_phc = np.concatenate(labels_phc_list, axis=0)
    y_amyloid = np.concatenate(labels_amyloid_list, axis=0)
    y_mci = np.concatenate(labels_mci_list, axis=0)
    y_age = np.concatenate(labels_age_list, axis=0)      
    y_gender = np.concatenate(labels_gender_list, axis=0) 

    # saving all the variables
    joblib.dump((X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender), cache_file) 

    return X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_enhanced_cm(y_true, y_pred, labels, task_name, save_path):
    # Calculate raw counts
    cm = confusion_matrix(y_true, y_pred)
    # Calculate percentages (normalized by row/true labels)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create text annotations: "Count\n(Percentage%)"
    annotations = np.array([["{0}\n({1:.1f}%)".format(count, perc) 
                             for count, perc in zip(row_count, row_perc)] 
                            for row_count, row_perc in zip(cm, cm_perc)])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix: {task_name}')
    plt.savefig(os.path.join(save_path, f"{task_name}_cm.png"), dpi=300)
    plt.close()

def regroup_dx_for_plot(y_dx):

    y_new = np.full_like(y_dx, -1)

    # CN + SMC → 0
    y_new[np.isin(y_dx, [0,1])] = 0

    # AD → 1
    y_new[y_dx == 2] = 1

    # LMCI + EMCI → 2
    y_new[np.isin(y_dx, [4,5])] = 2

    return y_new

def save_scatter_projection(X_2d, labels, save_path, title):

    class_info = {
        0: {"name": "CN + SMC", "color": "#1f77b4"},   # blue
        1: {"name": "AD",        "color": "#d62728"},  # red
        2: {"name": "LMCI + EMCI","color": "#2ca02c"}  # green
    }

    plt.figure(figsize=(6,5))

    for class_id in sorted(class_info.keys()):
        if class_id not in labels:
            continue

        idx = labels == class_id

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            color=class_info[class_id]["color"],
            label=class_info[class_id]["name"],
            s=20,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.3
        )

    plt.legend(frameon=True)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# downstream evaluation
def evaluate_downstream(X, y_dx_raw, y_phc_raw, y_amyloid_raw, y_mci_raw, y_age_raw, y_gender_raw, seed=42):
    results = {}
    preds = {}

    X_valid = X
    y_dx = y_dx_raw
    y_phc = y_phc_raw
    y_amyloid = y_amyloid_raw
    y_mci = y_mci_raw
    y_age = y_age_raw
    y_gender = y_gender_raw


    if not os.path.exists("rids_splits_cache"):
        guardar_rids_splits(y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender, SEEDS)

    def run_binary_task(X_task, y_task, task_name, results_dict, preds_dict):
        if np.sum(y_task != -1) > 20 and len(np.unique(y_task)) > 1:
            X_task = normalize(X_task)
            y_task = y_task.astype(int)

            #using seed
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_task, y_task, test_size=0.25, random_state=seed, stratify=y_task
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=1/3, random_state=seed, stratify=y_train_val
            )

            fracciones_log = np.logspace(np.log10(0.01), np.log10(0.25), num=4)
            fracciones_resto = [0.50, 0.75, 1.0]
            fracciones = np.concatenate((fracciones_log, fracciones_resto))
            for f in fracciones:
                prefix = f"{int(f*100)}%_"
                
                if f < 1.0:
                    X_train_frac, _, y_train_frac, _ = train_test_split(
                        X_train, y_train, train_size=f, random_state=seed, stratify=y_train
                    )
                else:
                    X_train_frac, y_train_frac = X_train, y_train

                if len(np.unique(y_train_frac)) < 2:
                    continue

                # validation is fixed
                X_cv = np.vstack((X_train_frac, X_val))
                y_cv = np.concatenate((y_train_frac, y_val))

                # index mask
                split_index = np.concatenate((
                    -1 * np.ones(len(X_train_frac)),
                    0 * np.ones(len(X_val))
                ))
                custom_split = PredefinedSplit(test_fold=split_index)

                pipe = make_pipeline(StandardScaler(), LogisticRegression(
                    solver="lbfgs", penalty="l2", max_iter=10000, class_weight="balanced"
                ))
                param_grid = {"logisticregression__C": np.logspace(-4, 2, 10)}
                
                # giving the predefined split to GridSearchCV within the parameter 'cv'
                grid = GridSearchCV(pipe, param_grid, cv=custom_split)
                grid.fit(X_cv, y_cv)  # trained on Train Frac and evaluated on X_val

                # predictions over the invariant test partition
                y_score_test = grid.predict_proba(X_test)[:, 1]
                y_pred_test = (y_score_test > 0.5).astype(int)

                # getting the metrics based on the percentage of data used in train
                results_dict[f'{prefix}{task_name}_b_acc'] = balanced_accuracy_score(y_test, y_pred_test)
                results_dict[f'{prefix}{task_name}_mcc'] = matthews_corrcoef(y_test, y_pred_test)
                results_dict[f'{prefix}{task_name}_auroc'] = roc_auc_score(y_test, y_score_test)
                
                preds_dict[f'{prefix}{task_name}'] = (y_test, y_score_test)

                # cloning the execution
                if f == 1.0:
                    results_dict[f'{task_name}_b_acc'] = results_dict[f'{prefix}{task_name}_b_acc']
                    results_dict[f'{task_name}_auroc'] = results_dict[f'{prefix}{task_name}_auroc']
                    preds_dict[task_name] = preds_dict[f'{prefix}{task_name}']

    #  AD vs CN/SMC
    mask1 = np.isin(y_dx, [0, 1, 2])
    if np.sum(mask1) > 20:
        run_binary_task(X_valid[mask1], np.where(y_dx[mask1] == 2, 1, 0), 'AD_vs_CN', results, preds)

    #  sMCI vs pMCI
    mask2 = y_mci != -1
    if np.sum(mask2) > 20:
        run_binary_task(X_valid[mask2], y_mci[mask2], 'sMCI_vs_pMCI', results, preds)

    #  Multiclase (CN vs MCI vs AD)
    mask3 = np.isin(y_dx, [0, 1, 2, 4, 5])
    if np.sum(mask3) > 30:
        X3 = normalize(X_valid[mask3])
        y3_raw = y_dx[mask3]
        y3_multi = np.zeros_like(y3_raw).astype(int)
        y3_multi[np.isin(y3_raw, [0, 1])] = 0   # CN
        y3_multi[np.isin(y3_raw, [4, 5])] = 1   # MCI
        y3_multi[y3_raw == 2] = 2               # AD

        X3_train_val, X3_test, y3_train_val, y3_test = train_test_split(
            X3, y3_multi, test_size=0.25, random_state=seed, stratify=y3_multi
        )
        X3_train, X3_val, y3_train, y3_val = train_test_split(
            X3_train_val, y3_train_val, test_size=1/3, random_state=seed, stratify=y3_train_val
        )
        fracciones_log = np.logspace(np.log10(0.01), np.log10(0.25), num=4)
        fracciones_resto = [0.50, 0.75, 1.0]
        fracciones = np.concatenate((fracciones_log, fracciones_resto))
        for f in fracciones:
            prefix = f"{int(f*100)}%_"
            if f < 1.0:
                X3_train_frac, _, y3_train_frac, _ = train_test_split(
                    X3_train, y3_train, train_size=f, random_state=seed, stratify=y3_train
                )
            else:
                X3_train_frac, y3_train_frac = X3_train, y3_train

            if len(np.unique(y3_train_frac)) < 3:
                continue

            pipe = make_pipeline(StandardScaler(), LogisticRegression(
                solver="lbfgs", penalty="l2", max_iter=10000, class_weight="balanced"
            ))
            min_samples = np.min(np.bincount(y3_train_frac))
            inner_cv = min(3, min_samples)

            if inner_cv < 2:
                grid = pipe.fit(X3_train_frac, y3_train_frac)
            else:
                grid = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 2, 8)}, cv=inner_cv)
                grid.fit(X3_train_frac, y3_train_frac)

            y3_score_test = grid.predict_proba(X3_test)
            y3_pred_test = np.argmax(y3_score_test, axis=1)

            results[f'{prefix}Multi_3Class_b_acc'] = balanced_accuracy_score(y3_test, y3_pred_test)
            results[f'{prefix}Multi_3Class_auroc'] = roc_auc_score(y3_test, y3_score_test, multi_class='ovr', average='macro')
            preds[f'{prefix}Multi_3Class'] = (y3_test, y3_score_test)

            if f == 1.0:
                results['Multi_3Class_b_acc'] = results[f'{prefix}Multi_3Class_b_acc']
                results['Multi_3Class_auroc'] = results[f'{prefix}Multi_3Class_auroc']
                preds['Multi_3Class'] = preds[f'{prefix}Multi_3Class']

    #  Regression PHC 
    mask4 = (y_phc != -1000.0) & (~np.isnan(y_phc))
    if np.sum(mask4) > 20:
        X4 = X_valid[mask4]
        y4 = y_phc[mask4]

        X4_train_val, X4_test, y4_train_val, y4_test = train_test_split(
            X4, y4, test_size=0.25, random_state=seed
        )
        X4_train, X4_val, y4_train, y4_val = train_test_split(
            X4_train_val, y4_train_val, test_size=1/3, random_state=seed
        )

        fracciones_log = np.logspace(np.log10(0.01), np.log10(0.25), num=4)
        fracciones_resto = [0.50, 0.75, 1.0]
        fracciones = np.concatenate((fracciones_log, fracciones_resto))
        for f in fracciones:
            prefix = f"{int(f*100)}%_"
            if f < 1.0:
                X4_train_frac, _, y4_train_frac, _ = train_test_split(
                    X4_train, y4_train, train_size=f, random_state=seed
                )
            else:
                X4_train_frac, y4_train_frac = X4_train, y4_train

            
            X4_cv = np.vstack((X4_train_frac, X4_val))
            y4_cv = np.concatenate((y4_train_frac, y4_val))

            split_index_reg = np.concatenate((
                -1 * np.ones(len(X4_train_frac)),
                0 * np.ones(len(X4_val))
            ))
            custom_split_reg = PredefinedSplit(test_fold=split_index_reg)

            pipe_reg = make_pipeline(StandardScaler(), Ridge())
            param_grid_reg = {"ridge__alpha": np.logspace(-3, 3, 10)}
            grid_reg = GridSearchCV(pipe_reg, param_grid_reg, cv=custom_split_reg)
            grid_reg.fit(X4_cv, y4_cv)
            
            y4_pred_test = grid_reg.predict(X4_test)
            results[f'{prefix}PHC_R2'] = r2_score(y4_test, y4_pred_test)
            preds[f'{prefix}PHC_Reg'] = (y4_test, y4_pred_test)

            if f == 1.0:
                results['PHC_R2'] = results[f'{prefix}PHC_R2']
                preds['PHC_Reg'] = preds[f'{prefix}PHC_Reg']

    #  Amyloid status
    mask5 = y_amyloid != -1
    if np.sum(mask5) > 20:
        run_binary_task(X_valid[mask5], y_amyloid[mask5], 'Amyloid', results, preds)

    #  Age Regression (no stratification)
    mask_age = (y_age > 0) & (~np.isnan(y_age))
    if np.sum(mask_age) > 20:
        X_age = X_valid[mask_age]
        y_a = y_age[mask_age]

        X_a_train_val, X_a_test, y_a_train_val, y_a_test = train_test_split(
            X_age, y_a, test_size=0.25, random_state=seed
        )
        X_a_train, X_a_val, y_a_train, y_a_val = train_test_split(
            X_a_train_val, y_a_train_val, test_size=1/3, random_state=seed
        )

        fracciones_log = np.logspace(np.log10(0.01), np.log10(0.25), num=4)
        fracciones_resto = [0.50, 0.75, 1.0]
        fracciones = np.concatenate((fracciones_log, fracciones_resto))

        for f in fracciones:
            prefix = f"{int(f*100)}%_"
            if f < 1.0:
                X_a_train_frac, _, y_a_train_frac, _ = train_test_split(
                    X_a_train, y_a_train, train_size=f, random_state=seed
                )
            else:
                X_a_train_frac, y_a_train_frac = X_a_train, y_a_train

            reg_age = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg_age.fit(X_a_train_frac, y_a_train_frac)
            
            y_a_pred_test = reg_age.predict(X_a_test)
            results[f'{prefix}Age_MAE'] = mean_absolute_error(y_a_test, y_a_pred_test)
            preds[f'{prefix}Age_Reg'] = (y_a_test, y_a_pred_test)

            if f == 1.0:
                results['Age_MAE'] = results[f'{prefix}Age_MAE']
                preds['Age_Reg'] = preds[f'{prefix}Age_Reg']

    #  Gender Classification
    mask_gender = y_gender != -1
    if np.sum(mask_gender) > 20:
        run_binary_task(X_valid[mask_gender], y_gender[mask_gender], 'Gender', results, preds)

    return results, preds, X_valid, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender

# visualization functions
def save_roc_comparison(all_model_preds, save_path, title="Task ROC Comparison"):
    plt.figure(figsize=(7, 7))
    for name, (y_true, y_score) in all_model_preds.items():
        if len(y_score.shape) > 1: continue 
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_auroc_bar_plot(results_dict, save_path):
    """ Grafica una comparativa de barras de AUROC filtrando las desviaciones estándar """
    df = pd.DataFrame(results_dict).T.reset_index().rename(columns={'index': 'Model'})
    
   
    cols = [c for c in df.columns if 'auroc' in c.lower() and ('mean' in c.lower() or 'multi' in c.lower())] + ['Model']
    df_melted = df[cols].melt(id_vars='Model', var_name='Task', value_name='AUROC')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Task', y='AUROC', hue='Model')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    plt.title("Downstream Tasks Performance (AUROC)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def compute_tsne_raw(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(X_scaled)

def compute_tsne_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(X_pca)

def save_confusion_matrix(y_true, y_pred, output_path, classes, title="Confusion Matrix"):
    try:
        y_true = np.array(y_true).astype(float)
        y_pred = np.array(y_pred).astype(float)
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask].astype(int)
        y_pred = y_pred[valid_mask].astype(int)

        if len(y_true) == 0: return
        labels = np.arange(len(classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title(title)
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f" Error en CM {output_path}: {e}")

def save_regression_plot(y_true, y_pred, save_path, title):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_tsne_plot(X_embedded, labels, title, filename, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=cmap, alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

def transform_with_fixed_umap(X, projection_dir):
    scaler = joblib.load(f"{projection_dir}/scaler.pkl")
    pca = joblib.load(f"{projection_dir}/pca.pkl")
    reducer = joblib.load(f"{projection_dir}/umap.pkl")
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return reducer.transform(X_pca)

def fit_and_save_umap_base(X_base, projection_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_pca)
    #joblib.dump(scaler, f"{projection_dir}/scaler.pkl")
    #joblib.dump(pca, f"{projection_dir}/pca.pkl")
    #joblib.dump(reducer, f"{projection_dir}/umap.pkl")
    return X_umap




def find_nii_files(root):
    nii_paths = []
    for dirpath, _, filenames in os.walk(root):
        if "M00" not in dirpath:
            continue
            
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            if fname.endswith(('.zip', '.tar.gz')):
                extracted_dir = extract_if_compressed(full_path)
                nii_paths.extend(find_nii_files(extracted_dir))
            elif fname.endswith(('.nii', '.nii.gz')):
                nii_paths.append(full_path)
    return nii_paths


def get_subject_id(path):
    match = re.search(r'(\d{3})S(\d{4})', str(path))
    if match: return f"{match.groups()[0]}_S_{match.groups()[1]}"
    return None

def get_last_available_epoch(checkpoint_dir, epochs_list):
    available = []
    for epoch in epochs_list:
        path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
        if os.path.exists(path):
            available.append(epoch)
    if not available:
        raise ValueError("No hay checkpoints disponibles.")
    return max(available)

class InferenceNiiDataset(Dataset):
    def __init__(self, dataframe, target_size):
        self.paths = dataframe['path'].tolist()
        self.labels_dx = dataframe['label_dx'].tolist()
        self.labels_phc = dataframe['label_phc'].tolist()
        self.labels_amyloid = dataframe['label_amyloid'].tolist() 
        self.labels_mci = dataframe['label_mci'].tolist()
        self.labels_age = dataframe['label_age'].tolist()
        self.labels_gender = dataframe['label_gender'].tolist()
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        dx = self.labels_dx[idx]
        phc = self.labels_phc[idx]
        
        img_nii = nib.load(path)
        img = img_nii.get_fdata().astype(np.float32) 
        img_nii.uncache() 

        if img.ndim == 4: img = img[..., 0] 
        img_tensor = torch.from_numpy(img).unsqueeze(0) 

        min_val, max_val = img_tensor.min(), img_tensor.max()
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        
        d, h, w = img_tensor.shape[1:]
        pad_d = max(0, self.target_size[0] - d)
        pad_h = max(0, self.target_size[1] - h)
        pad_w = max(0, self.target_size[2] - w)
        
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
        img_tensor = F.pad(img_tensor, padding, "constant", 0)
        img_tensor = img_tensor[:, :self.target_size[0], :self.target_size[1], :self.target_size[2]]
        
        amyloid = self.labels_amyloid[idx]
        mci = self.labels_mci[idx]
        age = self.labels_age[idx]
        gender = self.labels_gender[idx]
        
        return img_tensor, dx, phc, amyloid, mci, age, gender
    

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def save_multiclass_bar_plot(y_true, y_probs, output_path, epoch):
    """
    Genera una gráfica de barras de AUROC por clase (CN, MCI, AD).
    y_true: etiquetas (0, 1, 2)
    y_probs: matriz de probas (N, 3)
    """
    classes = ['CN', 'MCI', 'AD']
    n_classes = len(classes)
    
    # binarizing labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    aucs = []
    for i in range(n_classes):
        # individual auc of class i
        score = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        aucs.append(score)

    # design graph
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] # blue, green, orange
    
    bars = plt.bar(classes, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # adding the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, 1.1)
    plt.ylabel('AUROC', fontsize=12)
    plt.title(f'Task: AD vs MCI vs CN - Epoch {epoch}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_scatter_projection_amyloid(X_2d, labels, save_path, title):
    # defining the colors 0 blue negative and 1 red positive
    class_info = {
        0: {"name": "Amyloid Negative (0)", "color": "#1f77b4"},
        1: {"name": "Amyloid Positive (1)", "color": "#d62728"}
    }

    plt.figure(figsize=(6,5))

    for class_id in sorted(class_info.keys()):
        mask = labels == class_id
        if not np.any(mask):
            continue

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            color=class_info[class_id]["color"],
            label=class_info[class_id]["name"],
            s=20,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.3
        )

    plt.legend(frameon=True)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_experiment(experiment_name, model_type, inference_loader, device):
    print(f"\n{'='*60}")
    print(f" INICIANDO EXPERIMENTO: {experiment_name}")
    print(f" Arquitectura: {model_type}")
    print(f"{'='*60}\n")

    EXPERIMENT_NAME = experiment_name
    MODEL_TYPE = model_type
    CHECKPOINT_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/{EXPERIMENT_NAME}/checkpoints"
    RESULTS_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/results/{EXPERIMENT_NAME}_svc_lbfgs_AREPLICATE"
    EMBEDDINGS_DIR = f"{RESULTS_DIR}/embeddings_cache" 

    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOG_FILE = f"{RESULTS_DIR}/resumen_metricas_scarcity.txt"
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    PROJECTION_DIR = f"{RESULTS_DIR}/fixed_projection"
    os.makedirs(PROJECTION_DIR, exist_ok=True)

    EPOCHS_TO_PROCESS = [10, 50, 100, 200]
    
    COMPATIBLE_SIZE = (128, 160, 128) 

    if MODEL_TYPE == "resnet_simclr":
        CHECKPOINT_DIR = f"./simclr/checkpoints/{EXPERIMENT_NAME}"
    elif MODEL_TYPE == "vit_simclr_brainiac":
        CHECKPOINT_DIR = "./checkpoints/SimCLR_vit_base_brainiac"
    elif MODEL_TYPE == "simclr":
        CHECKPOINT_DIR = "./checkpoints/vit_databases_simclr_all_d/checkpoints"
    elif MODEL_TYPE == "resnet50":
        CHECKPOINT_DIR = "./checkpoints/SimCLR_resnet50_simclr"
    elif MODEL_TYPE == "resnet18":
        CHECKPOINT_DIR = "./checkpoints/SimCLR_resnet18_simclr"
    elif MODEL_TYPE == "ResNet34_3D_MAE":
        CHECKPOINT_DIR = F"./checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "UNETresnetL":
        CHECKPOINT_DIR = F"./checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "ResNet18_3D_MAE":
        CHECKPOINT_DIR = F"./checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "ResNet50_3D_MAE":
        CHECKPOINT_DIR = F"./checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"




    # main simplified loop
    with open(LOG_FILE, "w") as f:
        f.write("Epoch | AD_vs_CN_Acc | LMCI_vs_EMCI_Acc | Multi_3Class | PHC_R2 | Amyloid_Acc\n")

    last_epoch = get_last_available_epoch(CHECKPOINT_DIR, EPOCHS_TO_PROCESS)

    X_base, y_dx_base, y_phc_base, y_amyloid_base, y_mci_base, y_age_base, y_gender_base = get_or_extract_embeddings(
        last_epoch,
        inference_loader,
        MODEL_TYPE,
        f"{CHECKPOINT_DIR}/checkpoint_epoch_{last_epoch}.pth",
        EMBEDDINGS_DIR
    )

    valid_idx = y_dx_base != -1
    X_base_clean = X_base[valid_idx]

    X_umap_base = fit_and_save_umap_base(X_base_clean, PROJECTION_DIR)

    for epoch in EPOCHS_TO_PROCESS:

        checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth"
        if not os.path.exists(checkpoint_file):
            continue

        epoch_dir = f"{RESULTS_DIR}/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)

        X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender = get_or_extract_embeddings(
            epoch, inference_loader, MODEL_TYPE, checkpoint_file, EMBEDDINGS_DIR
        )

        for epoch in EPOCHS_TO_PROCESS:
            checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth"
            if not os.path.exists(checkpoint_file):
                continue
            epoch_dir = f"{RESULTS_DIR}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)

            # embedding extraction
            X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender = get_or_extract_embeddings(
                epoch, inference_loader, MODEL_TYPE, checkpoint_file, EMBEDDINGS_DIR
            )

            # random seeds
            lista_semillas = [42, 100, 134, 4332, 2026, 999, 29, 344, 283, 22, 43]

            if not os.path.exists("rids_splits_cache2"):
                guardar_rids_splits(y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender, lista_semillas)
            
            for seed in lista_semillas:
                print(f" -> Evaluando Downstream con Semilla: {seed}")
                metrics, preds, X_clean, y_clean_dx, y_clean_phc, y_clean_amyloid, y_clean_mci, y_clean_age, y_clean_gender = evaluate_downstream(
                    X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender, seed=seed
                )
                
                # saving independent json per seed
                ruta_json_seed = f"{epoch_dir}/metricas_epoch_{epoch}_seed_{seed}.json"
                with open(ruta_json_seed, "w") as f:
                    json.dump(metrics, f, indent=4)
# data preparation and experiment queue
import argparse
if __name__ == "__main__":
    
    USE_CLUSTER = 0 #if i am using or not the cluster to paral- (0 no 1 yes)

    if USE_CLUSTER == 1:
        # reading cluster configuration
        parser = argparse.ArgumentParser(description="Lanza el análisis downstream de un modelo específico.")
        parser.add_argument("--config", type=int, required=True, help="ID del experimento a ejecutar (0-6)")
        args = parser.parse_args()
        import traceback
        import torch
        if torch.cuda.is_available():
          
            major, minor = torch.cuda.get_device_capability()
            print(f" PASO 0: Verificando GPU... (Capacidad CUDA: {major}.{minor})")
            
            if major < 7:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f" ERROR GPU compatibility\n"
                    f"old GPU (Capacity {major}.{minor}).\n"
                    f"This PyTorch version requires a minimum capacity of 7.0.\n"
                    f"abort :(.\n"
                    f"{'='*60}"
                )
            else:
                print(" in cpu")

        print(f" Step 1: initializing the script for the ID config: {args.config}")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ROOT_DIR = "./databases/ADNI/M00"
        CSV_PATH = "./ADNIMERGE_25Aug2023.csv"
        AMYLOID_XLSX_PATH = "./analysis/code/downstream/UCBERKELEYAV45_amyloid_status.xlsx" 
        MCI_XLSX_PATH = "./analysis/code/downstream/Classification_MCI_to_AD.xlsx"
        PHC_XLSX_PATH = "./analysis/code/downstream/annual_memory_change_phc.xlsx"
        COMPATIBLE_SIZE = (128, 160, 128)
        BATCH_SIZE = 4

        try:
            print(" PASO 2: Cargando Excels y CSVs...")
            amyloid_df = pd.read_excel(AMYLOID_XLSX_PATH)
            rid_to_amyloid = dict(zip(amyloid_df["RID"].astype(int), amyloid_df["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"]))

            mci_df = pd.read_excel(MCI_XLSX_PATH)
            mci_map = {"Stable-MCI": 0, "Converter-MCI": 1}
            
            rid_to_mci = {int(row["RID"]): mci_map.get(row["CLASSIFICATION"], -1) 
                            for _, row in mci_df.dropna(subset=["RID"]).iterrows()}


            print(" Cargando Demográficos (Edad y Sexo)...")
          
            demographics_df = pd.read_csv(CSV_PATH, low_memory=False) 
            rid_to_age = dict(zip(demographics_df['RID'], demographics_df['AGE']))
            rid_to_gender = dict(zip(demographics_df['RID'], demographics_df['PTGENDER']))
            gender_map = {"Male": 0, "Female": 1}


            print(" Loading phc Excel...")
            phc_df = pd.read_excel(PHC_XLSX_PATH)
            # eliminating possible NaNs
            phc_df = phc_df.dropna(subset=["RID", "PHC_mem_rate"]) 
            rid_to_phc = dict(zip(phc_df["RID"].astype(int), phc_df["PHC_mem_rate"]))
            
            labels_df = pd.read_csv(CSV_PATH, low_memory=False)
            labels_df["PTID"] = labels_df["PTID"].str.strip()
            

            id_to_data = {}
            for _, row in labels_df.iterrows():
                id_to_data[row["PTID"]] = {"dx": row["DX_bl"]}

            diag_to_label = {"CN": 0, "SMC": 1, "AD": 2, "LMCI": 4, "EMCI": 5}

            print(" Step 3: Searching for NIfTIs...")
            all_nii = find_nii_files(ROOT_DIR)
            nii_info = []
            
            for path in all_nii:
                subj_id = get_subject_id(path) 
                if subj_id in id_to_data:
                    data = id_to_data[subj_id]
                    dx_label = diag_to_label.get(data["dx"], -1)
                    #phc_label = data["phc"]

                    match = re.search(r'(\d{3})S(\d{4})', str(path))
                    rid = int(match.groups()[1]) if match else -1
                    amyloid_label = rid_to_amyloid.get(rid, -1)

                    phc_label = rid_to_phc.get(rid, -1000.0)
                    
                    # mci extraction (label)
                    mci_label = rid_to_mci.get(rid, -1)
                    
                    age_label = rid_to_age.get(rid, -1.0)
                    gender_raw = rid_to_gender.get(rid, "Unknown")
                    gender_label = gender_map.get(gender_raw, -1)

                    nii_info.append((path, dx_label, phc_label, amyloid_label, mci_label, age_label, gender_label))

            print(f" Step 4: Validating NIfTIs...")
            valid_rows = []
            for path, dx, phc, amyloid, mci, age, gender in tqdm(nii_info):
                try:
                    nib.load(path)
                    valid_rows.append((path, dx, phc, amyloid, mci, age, gender))
                except Exception:
                    continue

            nii_df = pd.DataFrame(valid_rows, columns=["path", "label_dx", "label_phc", "label_amyloid", "label_mci", "label_age", "label_gender"])
            print(f" DataFrame listo con {len(nii_df)} sujetos válidos.")

            print(" Step 5: Creating the DataLoader...")
            inference_dataset = InferenceNiiDataset(nii_df, target_size=COMPATIBLE_SIZE)
            
            inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

            print("step 6: Selecting the model based on the configuration...")
            
            # the index defines the configuration
            lista_experimentos = [
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva1", "type": "vit_mae_3d"},         # Config 0
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_01", "type": "vit_mae_3d"},       # Config 1
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_05", "type": "vit_mae_3d"},       # Config 2
                {"name": "vit_databases_curriculum_ssim_decoder_v3_baseline_puro_mae", "type": "vit_mae_3d"},             # Config 3
                {"name": "resnet_databases_curriculum_ssim_v3real_ALL_d_formas", "type": "ResNet18_3D_MAE"},              # Config 4
                {"name": "vit_databases_curriculum_ssim_decoder_v3_ALL_d_fixed_cluster", "type": "vit_mae_all_d"},        # Config 5
                {"name": "vit_databases_curriculum_ssim_decoder_v3_ALL_d", "type": "vit_mae_all_d"},                      # Config 6
                {"name": "vit_databases_simclr_ALL_d", "type": "simclr"},                                                 # Config 7
                {"name": "resnet3d_simclr_paper", "type": "resnet_simclr"},                                               # Config 8
                {"name": "SimCLR_resnet18_simclr", "type": "resnet18"},                                                   # Config 9
                {"name": "SimCLR_resnet50_simclr", "type": "resnet50"},                                                   # Config 10
                {"name": "resnet_MAE_loss_SSIM", "type": "ResNet18_3D_MAE"},                                              # Config 11
                {"name": "vit_mae_block32_r8", "type": "vit_mae_small"},                                                  # Config 12
                {"name": "vit_mae_p16_ssim", "type": "vit_mae_base"},                                                     # Config 13
                {"name": "vit_databases_simclr_all_d", "type": "simclr"},                                                 # Config 14
                {"name": "vit_databases_simclr_all_d_base", "type": "simclr_base"},                                       # Config 15
                {"name": "resnet3d_simclr_paper_ALL_dtrue", "type": "resnet_simclr"},                                     # Config 16
                {"name": "SimCLR_vit_base_brainiac", "type": "vit_simclr_brainiac"},                                      # Config 17
                {"name": "resnet_MAE_loss_SSIM", "type": "ResNet18_3D_MAE"},                                              # Config 18
                {"name": "vit_mae_block32_r8", "type": "vit_mae_base"},                                                   # Config 19
                {"name": "resnet_databases_curriculum_ssim_v3real_ALL_dtrue2", "type": "ResNet18_3D_MAE"},                # Config 20
                {"name": "vit_databases_curriculum_ssim_decoder_v3_ALL_d", "type": "vit_mae_3d"},                         # Config 21
                {"name": "vit_mae_p16_msssim", "type": "vit_mae_small"},                                                  # Config 22
                {"name": "vit_mae_p16_mse", "type": "vit_mae_small"},                                                     # Config 23 

                {"name": "vit_databases_curriculum_ssim_decoder_v3_vit_base_mae_t0200_database_extended", "type": "vit_mae_base"},   # Config 24
                {"name": "vit_databases_curriculum_ssim_decoder_v3_vit_base_mae_t0200", "type": "vit_mae_base"},                                                     # Config 25 
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-64_t0200", "type": "ResNet18_3D_MAE"},                                                     # Config 26 
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-32_t0200", "type": "ResNet18_3D_MAE"},    
                {"name": "resnet_MAE_loss-ssim_mask-0.6_patch-64_t0200", "type": "ResNet18_3D_MAE"},    
                {"name": "resnet_MAE_loss-ssim_mask-0.6_patch-32_t0200", "type": "ResNet18_3D_MAE"},    
                {"name": "resnet_MAE_loss_SSIM_t0200", "type": "ResNet18_3D_MAE"},    #30
                {"name": "vit_mae_base_p16_ssim_3206", "type": "vit_mae_base"},
                {"name": "vit_mae_base_p16_ssim_3208", "type": "vit_mae_base"},
                {"name": "vit_mae_base_p16_ssim_6406", "type": "vit_mae_base"},
                {"name": "vit_mae_base_p16_ssim_6408", "type": "vit_mae_base"}, #34
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"}, #35
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"},
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"},
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"},
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"},
                {"name": "vit_databases_curriculum_ssim_decoder_v3_con_loss_contrastiva_001", "type": "vit_mae_3d"}, #40

                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-32_t0200_short", "type": "ResNet18_3D_MAE"}, #41
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-64_t0200_short", "type": "ResNet18_3D_MAE"},

                {"name": "resnet_MAE_loss_SSIM_t0200_database_extended", "type": "ResNet18_3D_MAE"},#43

                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-32_t0200_short_detail", "type": "ResNet18_3D_MAE"}, #44
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-64_t0200_short_detail", "type": "ResNet18_3D_MAE"},
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-32_t0200_short_detail", "type": "ResNet18_3D_MAE"}, #46
                {"name": "resnet_MAE_loss-ssim_mask-0.8_patch-64_t0200_short_detail", "type": "ResNet18_3D_MAE"},

                {"name": "resnet_MAE_loss-ssim_tissue-True_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"}, #48
                {"name": "resnet_MAE_loss-ssim_tissue-False_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"}, #49
                {"name": "resnet_MAE_loss-ssim_tissue-True_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"}, #50

                {"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"}, #51
                {"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"},
                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"},
                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"}, #54

                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"}, #55

                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symFalse", "type": "ResNet18_3D_MAE"}, #56
                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symTrue", "type": "ResNet18_3D_MAE"},

                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symTrue2", "type": "ResNet18_3D_MAE"}, #58
                {"name": "SimCLR_resnet18_simclr_short_det_loss_data", "type": "resnet18"}, #59
                {"name": "resnet3d_simclr_paper_ALL_dtrue", "type": "resnet_simclr"}, #60

                {"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-False_m-0.3_data", "type": "ResNet18_3D_MAE"}, #61
                {"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_data", "type": "ResNet18_3D_MAE"}, #62




                {"name": "2_local_test_medium_loss-ms-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},   #63                                              
                {"name": "2_local_test_medium_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                {"name": "2_resnet_MAE_M2_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                {"name": "2_UNETresnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "UNETresnetL"},

                {"name": "2_resnet_MAE_L_checkpoints_loss-ms-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},                                                 
                {"name": "2_resnet_MAE_L_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                {"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                {"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},
                
                {"name": "2_local_test_loss-ssim2_force_tissue-True_exclude_bg-True_curriculum_v3", "type": "ResNet18_3D_MAE"},
            ]
            

            # validation, managing errors
            if args.config < 0 or args.config >= len(lista_experimentos):
                raise ValueError(f"El ID proporcionado (--config {args.config}) no existe en la lista. Debe ser entre 0 y {len(lista_experimentos)-1}.")

            # selecting the chosen experiment
            exp_elegido = lista_experimentos[args.config]

            print(f"\n Running: {exp_elegido['name']} (Tipo: {exp_elegido['type']})\n")
            
            # running
            run_experiment(exp_elegido["name"], exp_elegido["type"], inference_loader, DEVICE)

        except Exception as e:
            print("\n" + "x"*20)
            print("EL SCRIPT HA CRASHEADO. ERROR:")
            traceback.print_exc()
            print("x"*20 + "\n")

    if USE_CLUSTER == 0:        
        import pandas as pd
        import torch
        import nibabel as nib
        import re
        from tqdm import tqdm
        from torch.utils.data import DataLoader
        import traceback

        # local configuration
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ROOT_DIR = "./databases/ADNI/M00"
        CSV_PATH = "./ADNIMERGE_25Aug2023.csv"
        AMYLOID_XLSX_PATH = "./analysis/code/downstream/UCBERKELEYAV45_amyloid_status.xlsx" 
        MCI_XLSX_PATH = "./analysis/code/downstream/Classification_MCI_to_AD.xlsx"
        PHC_XLSX_PATH = "./analysis/code/downstream/annual_memory_change_phc.xlsx"

        COMPATIBLE_SIZE = (128, 160, 128)
        BATCH_SIZE = 4 

    
        try:
            print(f" usando dispositivo: {DEVICE}")
            
            # excels and csv
            print(" PASO 1: Cargando etiquetas y datos demográficos...")
            
            # Amyloid
            amyloid_df = pd.read_excel(AMYLOID_XLSX_PATH)
            rid_to_amyloid = dict(zip(amyloid_df["RID"].astype(int), amyloid_df["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"]))

            # MCI
            mci_df = pd.read_excel(MCI_XLSX_PATH)
            mci_map = {"Stable-MCI": 0, "Converter-MCI": 1}
            rid_to_mci = {int(row["RID"]): mci_map.get(row["CLASSIFICATION"], -1) 
                            for _, row in mci_df.dropna(subset=["RID"]).iterrows()}

            # dem
            demographics_df = pd.read_csv(CSV_PATH, low_memory=False)
            rid_to_age = dict(zip(demographics_df['RID'], demographics_df['AGE']))
            rid_to_gender = dict(zip(demographics_df['RID'], demographics_df['PTGENDER']))
            gender_map = {"Male": 0, "Female": 1}

            # PHC
            phc_df = pd.read_excel(PHC_XLSX_PATH).dropna(subset=["RID", "PHC_mem_rate"])
            rid_to_phc = dict(zip(phc_df["RID"].astype(int), phc_df["PHC_mem_rate"]))
            
            # base diagnosis
            labels_df = demographics_df.copy() 
            labels_df["PTID"] = labels_df["PTID"].str.strip()
            id_to_data = {row["PTID"]: row["DX_bl"] for _, row in labels_df.iterrows()}
            diag_to_label = {"CN": 0, "SMC": 1, "AD": 2, "LMCI": 4, "EMCI": 5}

            # searching for niftis
            print(" Step 2: Finging NIfTI files...")
            all_nii = find_nii_files(ROOT_DIR)
            nii_info = []
            
            for path in all_nii:
                subj_id = get_subject_id(path) 
                if subj_id in id_to_data:
                    dx_raw = id_to_data[subj_id]
                    dx_label = diag_to_label.get(dx_raw, -1)

                    match = re.search(r'(\d{3})S(\d{4})', str(path))
                    rid = int(match.groups()[1]) if match else -1
                    
                    # label extraction
                    phc_label = rid_to_phc.get(rid, -1000.0)
                    amyloid_label = rid_to_amyloid.get(rid, -1)
                    mci_label = rid_to_mci.get(rid, -1)
                    age_label = rid_to_age.get(rid, -1.0)
                    gender_label = gender_map.get(rid_to_gender.get(rid, "Unknown"), -1)

                    nii_info.append((path, dx_label, phc_label, amyloid_label, mci_label, age_label, gender_label))

            # validating file integrity
            print(" Step 3: validating NIfTIs integrity...")
            valid_rows = []
            for row in tqdm(nii_info):
                path = row[0]
                try:
                    nib.load(path)
                    valid_rows.append(row)
                except Exception:
                    continue

            # DataFrame & DataLoader
            nii_df = pd.DataFrame(valid_rows, columns=[
                "path", "label_dx", "label_phc", "label_amyloid", 
                "label_mci", "label_age", "label_gender"
            ])
            print(f" Dataset listo: {len(nii_df)} sujetos.")

            inference_dataset = InferenceNiiDataset(nii_df, target_size=COMPATIBLE_SIZE)
            inference_loader = DataLoader(
                inference_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=2, 
                pin_memory=True if torch.cuda.is_available() else False
            )

            # running
            lista_experimentos = [
                #{"name": "1_CURRICULUM_L2_long_shapes_COSINE/ssim", "type": "ResNet50_3D_MAE"},
                {"name": "2_local_test_large_loss-ms-ssim-true_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                


                #{"name": "2_UNETresnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "UNETresnetL"},
                #{"name": "2_local_test_medium_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_resnet_MAE_M2_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                

                #{"name": "2_resnet_MAE_L_checkpoints_loss-ms-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},                                                 
                #{"name": "2_resnet_MAE_L_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                #{"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                #{"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},
                
                #{"name": "2_local_test_loss-ssim2_force_tissue-True_exclude_bg-True_curriculum_v3", "type": "ResNet18_3D_MAE"},
            ]
            num_cpus_local = 8
            def procesar_experimento_seguro(exp, loader, device):
                print(f"\n [CPU asignada] Iniciando modelo: {exp['name']}")
                try:
                    run_experiment(exp["name"], exp["type"], loader, device)
                except Exception as e:
                    print(f" Error en {exp['name']}: {e}")

            print(f"\nRunning {len(lista_experimentos)} parallel experiments using {num_cpus_local} CPUs...")
            
            
            Parallel(n_jobs=num_cpus_local)(
                delayed(procesar_experimento_seguro)(exp, inference_loader, DEVICE)
                for exp in lista_experimentos
            )

        except Exception:
            print("\n CRASH DEL SCRIPT:")
            traceback.print_exc()

