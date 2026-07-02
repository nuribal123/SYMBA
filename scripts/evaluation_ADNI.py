import os
import os
import getpass
import tempfile
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

from architectures import ResNet50_3D_UNet, ResNet34_3D_MAE, ResNet50_3D_MAE


def get_resnet18_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(BasicBlock3D, [2, 2, 2, 2], in_chans, proj_dim)

def get_resnet50_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(Bottleneck3D, [3, 4, 6, 3], in_chans, proj_dim)

def get_model(model_type, checkpoint_path, device):
    """Instancia el modelo completo (con decoder) y carga pesos."""
    
    # 1. Instanciación del modelo correcto
    if model_type == "ResNet18_3D_MAE":
        model = ResNet18_3D_MAE(in_chans=1)
        
    elif model_type == "ResNet34_3D_MAE":
        model = ResNet34_3D_MAE(in_chans=1)

    elif model_type == "ResNet50_3D_MAE":
        model = ResNet50_3D_MAE(in_chans=1)

    elif model_type == "UNETresnetL":
        model = ResNet50_3D_UNet(in_chans=1)

    elif model_type == "vit_mae_base":
        # CAMBIO: Usamos ViT3D_Autoencoder en lugar de la versión Inference
        model = ViT3D_Autoencoder(
            img_size=(128, 160, 128),
            patch_size=16, 
            embed_dim=768, 
            depth=12,
            num_heads=12,
            decoder_embed_dim=512, # Parámetros estándar del decoder MAE
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

    elif model_type == "resnet_simclr_contrastive" or model_type == "resnet_simclr_brainiac":
        model = get_resnet18_3d(in_chans=1, proj_dim=128)


    elif "simclr" in model_type:
        print(f"El modelo {model_type} es de tipo SimCLR y NO tiene decoder por defecto.")
        # Aquí podrías instanciar la clase SimCLR normal, pero no dará reconstrucción visual
        if "resnet18" in model_type:
            model = get_resnet18_3d(in_chans=1, proj_dim=128)

        else:
            #model = ViT3D_SimCLR_Base(img_size=(128, 160, 128), patch_size=16)
            model = ResNet50_3D_MAE(in_chans=1)  # Fallback a ResNet50_3D_MAE si no es resnet18

    else:
        raise ValueError(f"Modelo {model_type} no soportado para visualización.")

    # 2. Carga de Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if model_type == "resnet_simclr_contrastive":
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("projection_head")}
        

    # 3. FIX: Interpolación Genérica de pos_embed
    # Esto es vital si cambiaste el tamaño de imagen o patch entre entrenamiento e inferencia
    if 'pos_embed' in state_dict and hasattr(model, 'pos_embed'):
        pos_embed_checkpoint = state_dict['pos_embed']
        pos_embed_model = model.pos_embed
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"Adaptando pos_embed: {pos_embed_checkpoint.shape} -> {pos_embed_model.shape}")
            
            # Caso: Falta/Sobra token CLS
            if pos_embed_checkpoint.shape[1] == pos_embed_model.shape[1] - 1:
                cls_token_pos = pos_embed_model[:, :1, :]
                pos_embed_checkpoint = torch.cat((cls_token_pos, pos_embed_checkpoint), dim=1)
            elif pos_embed_checkpoint.shape[1] == pos_embed_model.shape[1] + 1:
                pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
            
            # Si después de ajustar el CLS siguen siendo distintos, interpolamos espacialmente
            if pos_embed_checkpoint.shape != pos_embed_model.shape:
                # Lógica de interpolación trilineal (ya la tenías bien, asegúrate de que se ejecute aquí)
                # ... (tu código de F.interpolate) ...
                state_dict['pos_embed'] = pos_embed_checkpoint # (después de interpolar)

    # 4. Carga final
    model.load_state_dict(state_dict, strict=False) 
    # Usamos strict=False por si el checkpoint tiene pesos del proyector SimCLR 
    # que el Autoencoder no necesita, o viceversa.
    
    return model.to(device).eval()

def extract_features_from_model(model, model_type, x):
    """Maneja las diferencias en los outputs de cada arquitectura."""
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
    elif model_type == "resnet_simclr_contrastive" or model_type == "resnet_simclr_brainiac":
        h, _ = model(x)
        return h
    elif model_type == "vit_mae_all_d":
        # ¡AQUÍ ESTÁ EL CAMBIO! Añadimos mask_size_voxels=16
        # Ratio 0.0 asegura que usamos la imagen completa para extraer features reales.
        _, _, z_global = model(x, mask_ratio=0.0, mask_size_voxels=16) 
        return z_global
    elif model_type == "vit_mae_base":
        # ¡AQUÍ ESTÁ EL CAMBIO! Añadimos mask_size_voxels=16
        # Ratio 0.0 asegura que usamos la imagen completa para extraer features reales.
        #_, _, z_global = model(x, mask_ratio=0.0, mask_size_voxels=16) 
        z_global = model(x)
        return z_global
    elif model_type == "vit_mae_small":
        z_global = model(x)
        return z_global
    elif model_type == "vit_mae_3d":
        # mask_ratio=0.0 es VITAL para que vea la imagen completa durante la evaluación
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
        return h  # Devolvemos la representación global h, no la proyección


# EXTRACCIÓN Y CACHÉ DE EMBEDDINGS
# Añade 'embeddings_dir' como último parámetro
def get_or_extract_embeddings(epoch, dataloader, model_type, checkpoint_path, embeddings_dir):
    cache_file = f"{embeddings_dir}/features_epoch_{epoch}_v4_amyloid_mci_age_gender.pkl" 
    
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    
    print(f"Extrayendo características para epoch {epoch}...")
    model = get_model(model_type, checkpoint_path, DEVICE)
    
    # Añadimos las listas para age y gender
    latent_vectors, labels_dx_list, labels_phc_list, labels_amyloid_list, labels_mci_list, labels_age_list, labels_gender_list = [], [], [], [], [], [], []
    
    with torch.no_grad():
        # Añadimos ages y genders al for
        for images, dxs, phcs, amyloids, mcis, ages, genders in tqdm(dataloader, leave=False):
            images = images.to(DEVICE)
            features = extract_features_from_model(model, model_type, images)
            
            latent_vectors.append(features.cpu().numpy())
            labels_dx_list.append(dxs.numpy())
            labels_phc_list.append(phcs.numpy())
            labels_amyloid_list.append(amyloids.numpy())
            labels_mci_list.append(mcis.numpy()) 
            # AÑADIR:
            labels_age_list.append(ages.numpy())
            labels_gender_list.append(genders.numpy())
            
    X = np.concatenate(latent_vectors, axis=0)
    y_dx = np.concatenate(labels_dx_list, axis=0)
    y_phc = np.concatenate(labels_phc_list, axis=0)
    y_amyloid = np.concatenate(labels_amyloid_list, axis=0)
    y_mci = np.concatenate(labels_mci_list, axis=0)
    y_age = np.concatenate(labels_age_list, axis=0)       # AÑADIDO
    y_gender = np.concatenate(labels_gender_list, axis=0) # AÑADIDO

    # Guardar TODAS las variables en el cache
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
        0: {"name": "CN + SMC", "color": "#1f77b4"},   # azul
        1: {"name": "AD",        "color": "#d62728"},  # rojo
        2: {"name": "LMCI + EMCI","color": "#2ca02c"}  # verde
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

# ======================================================
# 4. EVALUACIÓN DOWNSTREAM (TOTALMENTE INDEPENDIENTE)
# ======================================================
def evaluate_downstream(X, y_dx_raw, y_phc_raw, y_amyloid_raw, y_mci_raw, y_age_raw, y_gender_raw):
    results = {}
    preds = {}

    # SOLUCIÓN CRÍTICA 1: Mapear argumentos para evitar NameError o fugas desde el scope global
    X_valid = X
    y_dx = y_dx_raw
    y_phc = y_phc_raw
    y_amyloid = y_amyloid_raw
    y_mci = y_mci_raw
    y_age = y_age_raw
    y_gender = y_gender_raw

    cv_strat = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_reg = KFold(n_splits=4, shuffle=True, random_state=42)

    # Helper para clasificación binaria modificado para almacenar métricas por fold
    # Helper para clasificación binaria repetitiva con Std Dev
    def run_binary_task(X_task, y_task, task_name, results_dict, preds_dict):
        if np.sum(y_task != -1) > 20 and len(np.unique(y_task)) > 1:
            # Normalización por fila (L2) para embeddings
            X_task = normalize(X_task)
            y_task = y_task.astype(int) 
            
            # Arrays para guardar las predicciones completas (para plots ROC/CM)
            y_pred_total = np.zeros_like(y_task)
            y_score_total = np.zeros_like(y_task, dtype=float)

            # Listas para guardar el score de cada fold
            fold_b_acc, fold_mcc, fold_auroc = [], [], []

            for train_idx, test_idx in cv_strat.split(X_task, y_task):
                X_train, X_test = X_task[train_idx], X_task[test_idx]
                y_train, y_test = y_task[train_idx], y_task[test_idx]

                pipe = make_pipeline(StandardScaler(), LogisticRegression(
                    solver="lbfgs", penalty="l2", max_iter=10000, class_weight="balanced"
                ))
                grid = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 2, 10)}, cv=3)
                
                # Ajustar en train
                grid.fit(X_train, y_train)
                
                # Predecir en test
                y_score = grid.predict_proba(X_test)[:, 1]
                y_pred = grid.predict(X_test)
                
                # Guardar para los diccionarios globales
                y_score_total[test_idx] = y_score
                y_pred_total[test_idx] = y_pred
                
                # Evaluar el fold actual
                fold_b_acc.append(balanced_accuracy_score(y_test, y_pred))
                fold_mcc.append(matthews_corrcoef(y_test, y_pred))
                try:
                    fold_auroc.append(roc_auc_score(y_test, y_score))
                except ValueError:
                    pass # En caso de que un fold estratificado quede vacío de una clase por error
            
            # Almacenar Media y Desviación Estándar
            results_dict[f'{task_name}_b_acc_mean'] = np.mean(fold_b_acc)
            results_dict[f'{task_name}_b_acc_std'] = np.std(fold_b_acc)
            
            results_dict[f'{task_name}_mcc_mean'] = np.mean(fold_mcc)
            results_dict[f'{task_name}_mcc_std'] = np.std(fold_mcc)
            
            if fold_auroc:
                results_dict[f'{task_name}_auroc_mean'] = np.mean(fold_auroc)
                results_dict[f'{task_name}_auroc_std'] = np.std(fold_auroc)

            # Devolvemos las predicciones unidas para las gráficas
            preds_dict[task_name] = (y_task, y_score_total)

    # AD vs CN/SMC
    mask1 = np.isin(y_dx, [0, 1, 2])
    if np.sum(mask1) > 20:
        run_binary_task(X_valid[mask1], np.where(y_dx[mask1] == 2, 1, 0), 'AD_vs_CN', results, preds)

    # sMCI vs pMCI
    mask2 = y_mci != -1
    if np.sum(mask2) > 20:
        run_binary_task(X_valid[mask2], y_mci[mask2], 'sMCI_vs_pMCI', results, preds)

    # Multiclase (CN vs MCI vs AD)
    mask3 = np.isin(y_dx, [0, 1, 2, 4, 5])
    if np.sum(mask3) > 30:
        X3 = normalize(X_valid[mask3])
        y3_raw = y_dx[mask3]
        y3_multi = np.zeros_like(y3_raw).astype(int)
        y3_multi[np.isin(y3_raw, [0, 1])] = 0   # CN
        y3_multi[np.isin(y3_raw, [4, 5])] = 1   # MCI
        y3_multi[y3_raw == 2] = 2               # AD

        fold_b_acc, fold_auroc = [], []
        y3_pred_total = np.zeros_like(y3_multi)
        y3_score_total = np.zeros((len(y3_multi), 3))

        for train_idx, test_idx in cv_strat.split(X3, y3_multi):
            X_train, X_test = X3[train_idx], X3[test_idx]
            y_train, y_test = y3_multi[train_idx], y3_multi[test_idx]

            pipe = make_pipeline(StandardScaler(), LogisticRegression(
                solver="lbfgs", penalty="l2", max_iter=10000, class_weight="balanced"
            ))
            grid = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 2, 8)}, cv=3)
            grid.fit(X_train, y_train)

            fold_preds = grid.predict(X_test)
            y_score = grid.predict_proba(X_test)

            y3_pred_total[test_idx] = fold_preds
            y3_score_total[test_idx] = y_score

            fold_b_acc.append(balanced_accuracy_score(y_test, fold_preds))
            try:
                fold_auroc.append(roc_auc_score(y_test, y_score, multi_class='ovr', average='macro'))
            except ValueError:
                pass

        # Guardar en resultados
        results['Multi_3Class_b_acc_mean'] = np.mean(fold_b_acc)
        results['Multi_3Class_b_acc_std'] = np.std(fold_b_acc)
        results['Multi_3Class_auroc_mean'] = np.mean(fold_auroc)
        results['Multi_3Class_auroc_std'] = np.std(fold_auroc)
        
        preds['Multi_3Class'] = (y3_multi, y3_score_total)

    # Regresión PHC
    mask4 = (y_phc != -1000.0) & (~np.isnan(y_phc))
    if np.sum(mask4) > 20:
        X4 = X_valid[mask4]
        y4 = y_phc[mask4]
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        y4_pred = cross_val_predict(reg, X4, y4, cv=cv_reg)
        results['PHC_R2'] = r2_score(y4, y4_pred)
        preds['PHC_Reg'] = (y4, y4_pred)

    # Amyloid status
    mask5 = y_amyloid != -1
    if np.sum(mask5) > 20:
        run_binary_task(X_valid[mask5], y_amyloid[mask5], 'Amyloid', results, preds)

    # Age Regression adaptado a 4-folds con mean y std
    mask_age = (y_age > 0) & (~np.isnan(y_age))
    if np.sum(mask_age) > 20:
        X_age = X_valid[mask_age]
        y_a = y_age[mask_age]
        reg_age = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        
        fold_mae = []
        all_y_age_pred = np.zeros(len(y_a))
        
        for train_idx, val_idx in cv_reg.split(X_age, y_a):
            X_train, X_val = X_age[train_idx], X_age[val_idx]
            y_train, y_val = y_a[train_idx], y_a[val_idx]
            
            reg_age.fit(X_train, y_train)
            y_age_pred_fold = reg_age.predict(X_val)
            all_y_age_pred[val_idx] = y_age_pred_fold
            
            fold_mae.append(mean_absolute_error(y_val, y_age_pred_fold))
            
        results['Age_MAE_mean'] = np.mean(fold_mae)
        results['Age_MAE_std'] = np.std(fold_mae)
        preds['Age_Reg'] = (y_a, all_y_age_pred)

    # Gender Classification
    mask_gender = y_gender != -1
    if np.sum(mask_gender) > 20:
        run_binary_task(X_valid[mask_gender], y_gender[mask_gender], 'Gender', results, preds)

    return results, preds, X_valid, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender

# --- FUNCIONES DE VISUALIZACIÓN ---

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
    
    # SOLUCIÓN CRÍTICA 3: Filtrar para quedarme solo con las medias de AUROC o la tarea multiclase, evitando barras de STD inservibles
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
        print(f"Error en CM {output_path}: {e}")

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
    joblib.dump(scaler, f"{projection_dir}/scaler.pkl")
    joblib.dump(pca, f"{projection_dir}/pca.pkl")
    joblib.dump(reducer, f"{projection_dir}/umap.pkl")
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

# --- FUNCIÓN RECUPERADA QUE FALTABA ---
def get_subject_id(path):
    match = re.search(r'(\d{3})S(\d{4})', str(path))
    if match: return f"{match.groups()[0]}_S_{match.groups()[1]}"
    return None
# ---------------------------------------

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
        self.labels_amyloid = dataframe['label_amyloid'].tolist() # NUEVO
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
        
        amyloid = self.labels_amyloid[idx] # NUEVO
        mci = self.labels_mci[idx]
        amyloid = self.labels_amyloid[idx] 
        mci = self.labels_mci[idx]
        
        age = self.labels_age[idx]
        gender = self.labels_gender[idx]
        
        # DEVOLVER TAMBIÉN AGE Y GENDER
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
    
    # Binarizamos las etiquetas para calcular AUC por cada clase vs el resto
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    aucs = []
    for i in range(n_classes):
        # Calculamos el AUC individual de la clase i
        score = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        aucs.append(score)

    # Configuración estética de la gráfica
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] # Azul, Verde, Naranja (similar a la imagen)
    
    bars = plt.bar(classes, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Añadir los valores encima de las barras
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
    # Definimos colores: 0 = Negativo (Azul), 1 = Positivo (Rojo)
    class_info = {
        0: {"name": "Amyloid Negative (0)", "color": "#1f77b4"},
        1: {"name": "Amyloid Positive (1)", "color": "#d62728"}
    }

    plt.figure(figsize=(6,5))

    for class_id in sorted(class_info.keys()):
        mask = labels == class_id
        if not np.any(mask): # Si no hay sujetos de esta clase, la saltamos
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
    print(f"INICIANDO EXPERIMENTO: {experiment_name}")
    print(f"Arquitectura: {model_type}")
    print(f"{'='*60}\n")

    EXPERIMENT_NAME = experiment_name
    MODEL_TYPE = model_type
    CHECKPOINT_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/{EXPERIMENT_NAME}/checkpoints"
    RESULTS_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/results/{EXPERIMENT_NAME}_svc_lbfgs_AREPLICATE"
    EMBEDDINGS_DIR = f"{RESULTS_DIR}/embeddings_cache" 

    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOG_FILE = f"{RESULTS_DIR}/resumen_metricas.txt"
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    PROJECTION_DIR = f"{RESULTS_DIR}/fixed_projection"
    os.makedirs(PROJECTION_DIR, exist_ok=True)

    EPOCHS_TO_PROCESS = [600]

  
    COMPATIBLE_SIZE = (128, 160, 128) 

    if MODEL_TYPE == "resnet_simclr":
        CHECKPOINT_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/simclr/checkpoints/{EXPERIMENT_NAME}"
    elif MODEL_TYPE == "vit_simclr_brainiac":
        CHECKPOINT_DIR = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/checkpoints/SimCLR_vit_base_brainiac"
    elif MODEL_TYPE == "simclr":
        CHECKPOINT_DIR = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/vit_databases_simclr_all_d/checkpoints"
    elif MODEL_TYPE == "resnet50":
        CHECKPOINT_DIR = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/checkpoints/SimCLR_resnet50_simclr"
    elif MODEL_TYPE == "resnet18":
        CHECKPOINT_DIR = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/checkpoints/SimCLR_resnet18_simclr"
    elif MODEL_TYPE == "ResNet34_3D_MAE":
        #CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
        CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "UNETresnetL":
        CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "ResNet18_3D_MAE":
        #CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
        CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/{EXPERIMENT_NAME}/checkpoints"
    elif MODEL_TYPE == "ResNet50_3D_MAE":
        #CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/1_resnet_tensors_loss_and_model/{EXPERIMENT_NAME}/checkpoints"
        CHECKPOINT_DIR = F"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/checkpoints/{EXPERIMENT_NAME}/checkpoints"
    
    elif MODEL_TYPE == "resnet_simclr_contrastive":
        CHECKPOINT_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/simclr/checkpoints/{EXPERIMENT_NAME}"

    elif MODEL_TYPE == "resnet_simclr_brainiac":
        CHECKPOINT_DIR = f"/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/checkpoints/{EXPERIMENT_NAME}_final"





    # ======================================================
    # BUCLE PRINCIPAL SIMPLIFICADO
    # ======================================================
    with open(LOG_FILE, "w") as f:
        f.write("Epoch | AD_vs_CN_Acc | LMCI_vs_EMCI_Acc | Multi_3Class | PHC_R2 | Amyloid_Acc\n")

    last_epoch = get_last_available_epoch(CHECKPOINT_DIR, EPOCHS_TO_PROCESS)

    # AÑADIDO: y_mci_base al final
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

        # Embeddings (AÑADIDO: y_mci)
        X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender = get_or_extract_embeddings(
            epoch, inference_loader, MODEL_TYPE, checkpoint_file, EMBEDDINGS_DIR
        )

        # Downstream (AÑADIDO: y_clean_mci como retorno y y_mci como parámetro)
        metrics, preds, X_clean, y_clean_dx, y_clean_phc, y_clean_amyloid, y_clean_mci, y_clean_age, y_clean_gender = evaluate_downstream(
            X, y_dx, y_phc, y_amyloid, y_mci, y_age, y_gender
        )
        epoch_dir = f"{RESULTS_DIR}/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        joblib.dump(preds, f"{epoch_dir}/preds.pkl") 

        joblib.dump(y_clean_amyloid, f"{epoch_dir}/labels_amyloid.pkl")
        joblib.dump(y_clean_dx, f"{epoch_dir}/labels_dx.pkl")
        joblib.dump(y_clean_phc, f"{epoch_dir}/labels_phc.pkl")
        joblib.dump(y_clean_age, f"{epoch_dir}/labels_age.pkl")
        joblib.dump(y_clean_gender, f"{epoch_dir}/labels_gender.pkl")

        with open(f"{epoch_dir}/metricas_epoch_{epoch}.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
            # CONFUSION MATRICES
            if "AD_vs_CN" in preds:
                y_true, y_score = preds["AD_vs_CN"]
                y_pred = (y_score > 0.5).astype(int) # Convertir proba a clase 0/1
                save_confusion_matrix(y_true, y_pred, f"{epoch_dir}/cm_AD_vs_CN.png", 
                                    classes=['CN', 'AD'], 
                                    title=f"AD vs CN - Epoch {epoch}")
            
            if "sMCI_vs_pMCI" in preds:
                y_true, y_score = preds["sMCI_vs_pMCI"]
                y_pred = (y_score > 0.5).astype(int)
                save_confusion_matrix(y_true, y_pred, f"{epoch_dir}/cm_sMCI_vs_pMCI.png", 
                                    classes=['sMCI', 'pMCI'], 
                                    title=f"sMCI vs pMCI - Epoch {epoch}")

            if "Multi_3Class" in preds:
                y_true, y_score = preds["Multi_3Class"]
                y_pred = np.argmax(y_score, axis=1) # Elegir la clase con mayor proba
                save_confusion_matrix(y_true, y_pred, f"{epoch_dir}/cm_Multiclass.png", 
                                    classes=['CN', 'MCI', 'AD'], 
                                    title=f"Multiclass - Epoch {epoch}")

                # 2. Para la Gráfica de Barras usamos las probabilidades (y_score)
                save_multiclass_bar_plot(y_true, y_score, f"{epoch_dir}/bar_auroc_multiclass.png", epoch)

            if "PHC_Reg" in preds:
                y_true, y_pred = preds["PHC_Reg"]
                save_regression_plot(y_true, y_pred, f"{epoch_dir}/regression_phc.png", 
                                     f"phc Regression - Epoch {epoch}")
                
            if "Amyloid" in preds:
                y_true, y_score = preds["Amyloid"]
                y_pred = (y_score > 0.5).astype(int)
                save_confusion_matrix(y_true, y_pred, f"{epoch_dir}/cm_Amyloid.png", 
                                    classes=['Neg', 'Pos'], 
                                    title=f"Amyloid - Epoch {epoch}")                
            if "task_age" in preds:
                y_true, y_pred = preds["task_age"]
                save_regression_plot(y_true, y_pred, f"{epoch_dir}/regression_age.png", 
                                     f"Age Regression - Epoch {epoch}")

        print(f"Epoch {epoch} | "
            f"AD/CN: {metrics.get('AD_vs_CN_b_acc',0):.3f} | "
            f"MCI: {metrics.get('LMCI_vs_EMCI_b_acc',0):.3f} | "
            f"3C: {metrics.get('Multi_3Class_b_acc',0):.3f} | "
            f"R2: {metrics.get('PHC_R2',0):.3f} | "
            f"Amy: {metrics.get('Amyloid_Acc',0):.3f}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch} | {metrics.get('AD_vs_CN_b_acc',0):.4f} | {metrics.get('LMCI_vs_EMCI_b_acc',0):.4f} | {metrics.get('Multi_3Class_b_acc',0):.4f} | {metrics.get('PHC_R2',0):.4f} | {metrics.get('Amyloid_Acc',0):.4f}\n")

        if X_clean.shape[0] < 50:
            continue

        # Proyecciones
        X_tsne_raw = compute_tsne_raw(X_clean)
        X_tsne_pca = compute_tsne_pca(X_clean)
        X_umap = transform_with_fixed_umap(X_clean, PROJECTION_DIR) # <--- AÑADIDO

        joblib.dump(X_tsne_raw, f"{epoch_dir}/tsne_raw.pkl")
        joblib.dump(X_tsne_pca, f"{epoch_dir}/tsne_pca.pkl")
        joblib.dump(X_umap, f"{epoch_dir}/umap_fixed.pkl")
        
        # GUARDAR FIGURAS EMBEDDING (DIAGNÓSTICO)
        y_plot = regroup_dx_for_plot(y_clean_dx)
        save_scatter_projection(X_tsne_raw, y_plot, f"{epoch_dir}/tsne_raw_dx.png", f"t-SNE Raw (DX) - Epoch {epoch}")
        save_scatter_projection(X_tsne_pca, y_plot, f"{epoch_dir}/tsne_pca_dx.png", f"t-SNE PCA (DX) - Epoch {epoch}")
        save_scatter_projection(X_umap, y_plot, f"{epoch_dir}/umap_fixed_dx.png", f"UMAP Fixed (DX) - Epoch {epoch}")
        all_preds = {
            "NeuroFM": joblib.load(f"{epoch_dir}/preds.pkl")["AD_vs_CN"],
            "BrainIAC*": joblib.load(f"{epoch_dir}/preds.pkl")["AD_vs_CN"]
        }
        save_roc_comparison(all_preds, f"{epoch_dir}/comparativa_roc.png")

        # Uso:
        X_tsne = TSNE(n_components=2).fit_transform(X)

        # t-SNE por Edad (Escala continua)
        X_tsne_clean = compute_tsne_pca(X_clean) # Mejor usar la versión con PCA para que sea rápido
        X_umap_clean = transform_with_fixed_umap(X_clean, PROJECTION_DIR)

        # t-SNE por Edad (Usa y_clean_age, no y_age)
        save_tsne_plot(X_tsne_clean, y_clean_age, f"t-SNE Age - Ep {epoch}", f"{epoch_dir}/tsne_age.png")

        # t-SNE por Sexo
        save_tsne_plot(X_tsne_clean, y_clean_gender, f"t-SNE Gender - Ep {epoch}", f"{epoch_dir}/tsne_gender.png", cmap='coolwarm')

        # Log de métricas (Asegúrate de que coincidan las llaves)
        print(f"Epoch {epoch} | "
              f"AD/CN: {metrics.get('AD_vs_CN_b_acc',0):.3f} | "
              f"MCI: {metrics.get('sMCI_vs_pMCI_b_acc',0):.3f} | "
              f"Age MAE: {metrics.get('Age_MAE',0):.2f}")

        # GUARDAR FIGURAS EMBEDDING (AMYLOID)
        valid_amyloid = y_clean_amyloid != -1
        if np.sum(valid_amyloid) > 0:
            y_amy_plot = y_clean_amyloid[valid_amyloid]
            save_scatter_projection_amyloid(X_tsne_raw[valid_amyloid], y_amy_plot, f"{epoch_dir}/tsne_raw_amyloid.png", f"t-SNE Raw (Amyloid) - Epoch {epoch}")
            save_scatter_projection_amyloid(X_tsne_pca[valid_amyloid], y_amy_plot, f"{epoch_dir}/tsne_pca_amyloid.png", f"t-SNE PCA (Amyloid) - Epoch {epoch}")
            save_scatter_projection_amyloid(X_umap[valid_amyloid], y_amy_plot, f"{epoch_dir}/umap_fixed_amyloid.png", f"UMAP Fixed (Amyloid) - Epoch {epoch}")


# ======================================================
# PREPARACIÓN DE DATOS Y COLA DE EXPERIMENTOS
# ======================================================
import argparse
if __name__ == "__main__":
    
    USE_CLUSTER = 0 #if i am using or not the cluster to paral- (0 no 1 yes)

    if USE_CLUSTER == 1:
        # --- NUEVO: Leer argumento del clúster ---
        parser = argparse.ArgumentParser(description="Lanza el análisis downstream de un modelo específico.")
        parser.add_argument("--config", type=int, required=True, help="ID del experimento a ejecutar (0-6)")
        args = parser.parse_args()
        # -----------------------------------------
        import traceback

        import torch
        if torch.cuda.is_available():
            # get_device_capability devuelve una tupla, ej: (6, 1) para la 1080 Ti
            major, minor = torch.cuda.get_device_capability()
            print(f"PASO 0: Verificando GPU... (Capacidad CUDA: {major}.{minor})")
            
            if major < 7:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"ERROR FATAL DE COMPATIBILIDAD DE GPU\n"
                    f"El clúster ha asignado una GPU demasiado antigua (Capacidad {major}.{minor}).\n"
                    f"Esta versión de PyTorch requiere una capacidad mínima de 7.0.\n"
                    f"Abortando la ejecución INMEDIATAMENTE para no malgastar tiempo.\n"
                    f"{'='*60}"
                )
            else:
                print("PASO 0: No se detectó GPU. Ejecutando en CPU.")

        print(f"PASO 1: Iniciando script para el config ID: {args.config}")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ROOT_DIR = "/export/data_ml4ds/Neurocosas/databases/ADNI/M00"
        CSV_PATH = "/export/usuarios01/nbesteban/ADNIMERGE_25Aug2023.csv"
        AMYLOID_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/UCBERKELEYAV45_amyloid_status.xlsx" 
        MCI_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/Classification_MCI_to_AD.xlsx"
        PHC_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/annual_memory_change_phc.xlsx"
        COMPATIBLE_SIZE = (128, 160, 128)
        BATCH_SIZE = 4

        try:
            print("PASO 2: Cargando Excels y CSVs...")
            amyloid_df = pd.read_excel(AMYLOID_XLSX_PATH)
            rid_to_amyloid = dict(zip(amyloid_df["RID"].astype(int), amyloid_df["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"]))

            # --- NUEVO: Cargar MCI y mapear a binario ---
            mci_df = pd.read_excel(MCI_XLSX_PATH)
            mci_map = {"Stable-MCI": 0, "Converter-MCI": 1}
            # Creo un diccionario RID -> 0 o 1. Si no existe o tiene otro valor, guardará -1
            rid_to_mci = {int(row["RID"]): mci_map.get(row["CLASSIFICATION"], -1) 
                            for _, row in mci_df.dropna(subset=["RID"]).iterrows()}
            # --------------------------------------------

            print("Cargando Demográficos (Edad y Sexo)...")
            # CUIDADO: ¡ADNIMERGE es un CSV, no un Excel! Usa read_csv
            demographics_df = pd.read_csv(CSV_PATH, low_memory=False) 
            rid_to_age = dict(zip(demographics_df['RID'], demographics_df['AGE']))
            rid_to_gender = dict(zip(demographics_df['RID'], demographics_df['PTGENDER']))
            gender_map = {"Male": 0, "Female": 1}
            # =========================

            print("Cargando Excel de PHC_mem_rate...")
            phc_df = pd.read_excel(PHC_XLSX_PATH)
            # Limpiamos posibles NaNs en la variable objetivo o RID
            phc_df = phc_df.dropna(subset=["RID", "PHC_mem_rate"]) 
            rid_to_phc = dict(zip(phc_df["RID"].astype(int), phc_df["PHC_mem_rate"]))
            
            labels_df = pd.read_csv(CSV_PATH, low_memory=False)
            labels_df["PTID"] = labels_df["PTID"].str.strip()
            

            id_to_data = {}
            for _, row in labels_df.iterrows():
                id_to_data[row["PTID"]] = {"dx": row["DX_bl"]}

            diag_to_label = {"CN": 0, "SMC": 1, "AD": 2, "LMCI": 4, "EMCI": 5}

            print("PASO 3: Buscando NIfTIs...")
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
                    
                    # --- NUEVO: Extraer etiqueta MCI ---
                    mci_label = rid_to_mci.get(rid, -1)
                    # -----------------------------------
                    
                    age_label = rid_to_age.get(rid, -1.0)
                    gender_raw = rid_to_gender.get(rid, "Unknown")
                    gender_label = gender_map.get(gender_raw, -1)

                    # Añadir a nii_info (recuerda actualizar el orden si es necesario)
                    nii_info.append((path, dx_label, phc_label, amyloid_label, mci_label, age_label, gender_label))

            print(f"Validating NIfTIs...")
            valid_rows = []
            for path, dx, phc, amyloid, mci, age, gender in tqdm(nii_info):
                try:
                    nib.load(path)
                    valid_rows.append((path, dx, phc, amyloid, mci, age, gender))
                except Exception:
                    continue

            # --- ACTUALIZADO: Añadir 'label_mci' a las columnas del DataFrame ---
            nii_df = pd.DataFrame(valid_rows, columns=["path", "label_dx", "label_phc", "label_amyloid", "label_mci", "label_age", "label_gender"])
            print(f"DataFrame listo con {len(nii_df)} sujetos válidos.")
            import re
            import os
            
            # 1. Función para extraer el RID de la columna 'path' de forma segura
            def extraer_rid(ruta):
                match = re.search(r'(\d{3})S(\d{4})', str(ruta))
                return int(match.groups()[1]) if match else -1
                
            nii_df['rid'] = nii_df['path'].apply(extraer_rid)

            # 2. Crear y guardar el mapeo (vincula la posición del DataFrame/pkl al RID)
            mapeo_rid_df = nii_df[["rid"]].reset_index().rename(columns={"index": "posicion_pkl", "rid": "rid_real"})
            
            os.makedirs("rids_splits_cache", exist_ok=True)
            ruta_mapeo_maestro = "rids_splits_cache/mapeo_maestro_posicion_a_rid.csv"
            mapeo_rid_df.to_csv(ruta_mapeo_maestro, index=False)
            print(f"Mapeo maestro guardado con éxito en: {ruta_mapeo_maestro}")

            

            print("Creating DataLoader...")
            inference_dataset = InferenceNiiDataset(nii_df, target_size=COMPATIBLE_SIZE)
            
            # Recuerda mantener num_workers=0 para evitar los crasheos de 8 segundos
            inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

            print("Choosing el modelo según el config...")
            
            # Tu lista actúa como el diccionario. El índice es el número del --config
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
            

            # Validación de seguridad: Comprobar que el número del clúster existe en la lista
            if args.config < 0 or args.config >= len(lista_experimentos):
                raise ValueError(f"El ID proporcionado (--config {args.config}) no existe en la lista. Debe ser entre 0 y {len(lista_experimentos)-1}.")

            # Seleccionamos ÚNICAMENTE el experimento que nos ha pedido el clúster
            exp_elegido = lista_experimentos[args.config]

            print(f"\nEJECUTANDO ÚNICO TRABAJO: {exp_elegido['name']} (Tipo: {exp_elegido['type']})\n")
            
            # Lanzamos la función
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

        # --- CONFIGURACIÓN LOCAL ---
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ROOT_DIR = "/export/data_ml4ds/Neurocosas/databases/ADNI/M00"
        CSV_PATH = "/export/usuarios01/nbesteban/ADNIMERGE_25Aug2023.csv"
        AMYLOID_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/UCBERKELEYAV45_amyloid_status.xlsx" 
        MCI_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/Classification_MCI_to_AD.xlsx"
        PHC_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/annual_memory_change_phc.xlsx"

        COMPATIBLE_SIZE = (128, 160, 128)
        BATCH_SIZE = 4  # Ajustado para no saturar la GPU local

    
        try:
            print(f" usando dispositivo: {DEVICE}")
            
            # 1. Cargar Excels y CSVs
            print("Cargando etiquetas y datos demográficos...")
            
            # Amiloide
            amyloid_df = pd.read_excel(AMYLOID_XLSX_PATH)
            rid_to_amyloid = dict(zip(amyloid_df["RID"].astype(int), amyloid_df["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"]))

            # MCI
            mci_df = pd.read_excel(MCI_XLSX_PATH)
            mci_map = {"Stable-MCI": 0, "Converter-MCI": 1}
            rid_to_mci = {int(row["RID"]): mci_map.get(row["CLASSIFICATION"], -1) 
                            for _, row in mci_df.dropna(subset=["RID"]).iterrows()}

            # Demográficos (ADNIMERGE es CSV)
            demographics_df = pd.read_csv(CSV_PATH, low_memory=False)
            rid_to_age = dict(zip(demographics_df['RID'], demographics_df['AGE']))
            rid_to_gender = dict(zip(demographics_df['RID'], demographics_df['PTGENDER']))
            gender_map = {"Male": 0, "Female": 1}

            # PHC
            phc_df = pd.read_excel(PHC_XLSX_PATH).dropna(subset=["RID", "PHC_mem_rate"])
            rid_to_phc = dict(zip(phc_df["RID"].astype(int), phc_df["PHC_mem_rate"]))
            
            # Diagnóstico base
            labels_df = demographics_df.copy() # Usamos el mismo CSV cargado
            labels_df["PTID"] = labels_df["PTID"].str.strip()
            id_to_data = {row["PTID"]: row["DX_bl"] for _, row in labels_df.iterrows()}
            diag_to_label = {"CN": 0, "SMC": 1, "AD": 2, "LMCI": 4, "EMCI": 5}

            # 2. Buscar y cruzar NIfTIs
            print("Buscando y cruzando archivos NIfTI...")
            all_nii = find_nii_files(ROOT_DIR)
            nii_info = []
            
            for path in all_nii:
                subj_id = get_subject_id(path) 
                if subj_id in id_to_data:
                    dx_raw = id_to_data[subj_id]
                    dx_label = diag_to_label.get(dx_raw, -1)

                    match = re.search(r'(\d{3})S(\d{4})', str(path))
                    rid = int(match.groups()[1]) if match else -1
                    
                    # Extraer todas las etiquetas
                    phc_label = rid_to_phc.get(rid, -1000.0)
                    amyloid_label = rid_to_amyloid.get(rid, -1)
                    mci_label = rid_to_mci.get(rid, -1)
                    age_label = rid_to_age.get(rid, -1.0)
                    gender_label = gender_map.get(rid_to_gender.get(rid, "Unknown"), -1)

                    nii_info.append((path, dx_label, phc_label, amyloid_label, mci_label, age_label, gender_label))

            # 3. Validar integridad de archivos
            print("Validating NIfTIs integrity...")
            valid_rows = []
            for row in tqdm(nii_info):
                path = row[0]
                try:
                    nib.load(path)
                    valid_rows.append(row)
                except Exception:
                    continue

            # 4. Crear DataFrame y DataLoader
            nii_df = pd.DataFrame(valid_rows, columns=[
                "path", "label_dx", "label_phc", "label_amyloid", 
                "label_mci", "label_age", "label_gender"
            ])
            print(f"Dataset listo: {len(nii_df)} sujetos.")
            import re
            import os
            
            # 1. Función para extraer el RID de la columna 'path' de forma segura
            def extraer_rid(ruta):
                match = re.search(r'(\d{3})S(\d{4})', str(ruta))
                return int(match.groups()[1]) if match else -1
                
            nii_df['rid'] = nii_df['path'].apply(extraer_rid)

            # 2. Crear y guardar el mapeo (vincula la posición del DataFrame/pkl al RID)
            mapeo_rid_df = nii_df[["rid"]].reset_index().rename(columns={"index": "posicion_pkl", "rid": "rid_real"})
            
            os.makedirs("rids_splits_cache", exist_ok=True)
            ruta_mapeo_maestro = "rids_splits_cache/mapeo_maestro_posicion_a_rid.csv"
            mapeo_rid_df.to_csv(ruta_mapeo_maestro, index=False)
            print(f" Mapeo maestro guardado con éxito en: {ruta_mapeo_maestro}")


            inference_dataset = InferenceNiiDataset(nii_df, target_size=COMPATIBLE_SIZE)
            inference_loader = DataLoader(
                inference_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=0, # 0 es más seguro para local en Windows
                pin_memory=True if torch.cuda.is_available() else False
            )

            # 5. Ejecutar lista de experimentos
            lista_experimentos = [

                {"name": "resnet3d_simclr_paper_ALL_dtrue_short_det_loss_data_all_d", "type": "resnet_simclr_contrastive"},
                
                #BRAIN SPECIFIC
                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symFalse", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symTrue2", "type": "ResNet18_3D_MAE"},
                ###{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symFalse", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-False_m-0.3_data", "type": "ResNet18_3D_MAE"},

                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"},

                #{"name": "1_CURRICULUM_L2_long_COSINE/ssim", "type": "ResNet50_3D_MAE"},
                #{"name": "1_CURRICULUM_L2_long_shapes_COSINE/ssim", "type": "ResNet50_3D_MAE"},


                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symTrue2", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3_symFalse", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-True2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-False_m-0.3", "type": "ResNet18_3D_MAE"},
                #{"name": "resnet_MAE_loss-ssim_tissue-False2_noBG-True_m-0.3", "type": "ResNet18_3D_MAE"},
                #{"name": "2_local_test_large_loss-ms-ssim-true_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},

                
                #{"name": "resnet3d_simclr_paper_ALL_dtrue_short_det_loss_data", "type": "resnet_simclr_contrastive"},
                #{"name": "SimCLR_resnet18_simclr_short_det_loss_data", "type": "resnet_simclr_brainiac"},
                
                #{"name": "1_CURRICULUM_S2_long_shapes/ssim", "type": "ResNet18_3D_MAE"},
                #{"name": "1_CURRICULUM_S2_long_age/ssim_w0.001", "type": "ResNet18_3D_MAE"},
                #{"name": "1_CURRICULUM_S2_long_age/ssim_w0.01", "type": "ResNet18_3D_MAE"},
                #{"name": "1_CURRICULUM_S2_long/ssim", "type": "ResNet18_3D_MAE"},
                #{"name": "1_CURRICULUM_M2_long/ssim", "type": "ResNet34_3D_MAE"},
                #CURRICULUM Y DESPUÉS FIJO :)

                
                #{"name": "2_local_test_medium_loss-ms-ssim-true_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_local_test_medium_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_local_test_medium_loss-ms-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_local_test_medium_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_resnet_MAE_M2_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet34_3D_MAE"},
                #{"name": "2_UNETresnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "UNETresnetL"},

                #{"name": "2_resnet_MAE_L_checkpoints_loss-ms-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},                                                 
                #{"name": "2_resnet_MAE_L_checkpoints_loss-mse_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                #{"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", "type": "ResNet50_3D_MAE"},
                #{"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-32_symTrue", "type": "ResNet50_3D_MAE"},
                
                #{"name": "2_local_test_loss-ssim2_force_tissue-True_exclude_bg-True_curriculum_v3", "type": "ResNet18_3D_MAE"},

                
            ]

            for exp in lista_experimentos:
                print(f"\n Initializing: {exp['name']}")
                try:
                    run_experiment(exp["name"], exp["type"], inference_loader, DEVICE)
                except Exception as e:
                    print(f"Error in {exp['name']}: {e}")
                    continue

        except Exception:
            print("\n CRASH OF THE SCRIPT:")
            traceback.print_exc()




