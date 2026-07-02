import os
import os
import getpass
import tempfile
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_predict, PredefinedSplit
import copy
# Get your cluster username
username = getpass.getuser()

# Create a user-specific cache directory
cache_dir = os.path.join(tempfile.gettempdir(), f"{username}_numba_cache")

# Create the directory if it doesn't exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

os.environ['NUMBA_CACHE_DIR'] = cache_dir
from sklearn.metrics import roc_auc_score, roc_curve, auc
import random
import numpy as np
import torch
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
sys.path.append("/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/training_code/MAE")
from resnet_databases_curriculum_ssim import ResNet3D_Autoencoder
sys.path.append("/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/training_code/MAE/all_d/classes")
from ViT3D_Autoencoder import ViT3D_Autoencoder_all_d
sys.path.append("/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/architectures_amyloid")
from architectures import ResNet_3D_SimCLR, BasicBlock3D, Bottleneck3D, ViT3D_SimCLR, ViT3D_MAE_Base_Inference, SimCLR_ViT3D_BrainIAC, ResNet18_3D_MAE, ViT3D_SimCLR_Base, PatchEmbed3D, PatchEmbed3D_BrainIAC, ViT3D_Autoencoder_age
sys.path.append("/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/modules")
from visualization_functions import extract_if_compressed, find_nii_files, get_subject_id, get_last_available_epoch, preprocess_image, mask_patches_fixed, save_comparison_slices, get_visualization_brain, save_reconstruction_plot 
from visualization_architectures import ResNet50_3D_UNet, ResNet34_3D_MAE, ResNet50_3D_MAE
from sklearn.metrics import roc_auc_score, r2_score


def set_all_seeds(seed):
    """Establece las semillas para garantizar reproducibilidad en cada corrida."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Opcional: Forzar determinismo en operaciones de GPU (puede ralentizar un poco)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_resnet18_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(BasicBlock3D, [2, 2, 2, 2], in_chans, proj_dim)

def get_resnet50_3d(in_chans=1, proj_dim=128):
    return ResNet_3D_SimCLR(Bottleneck3D, [3, 4, 6, 3], in_chans, proj_dim)

def get_model(model_type, checkpoint_path=None, device="cuda", pretrained=True):
    """
    Inicializa la arquitectura. 
    Si pretrained=True, carga los pesos del MAE de tu entrenamiento por curriculum.
    Si pretrained=False, devuelve la red con inicialización aleatoria (Desde cero).
    """
    # 1. Importamos dinámicamente las clases de script original de curriculum
    
    
    print(f" Inicializando arquitectura: {model_type}")
    
    # 2. Instanciar según el tipo de tu modelo
    if model_type == "ResNet50_3D_MAE":
        # Asegúrate de usar los mismos parámetros de parche que en tu curriculum
        model = ResNet50_3D_MAE(in_chans=1)
        embed_dim = 2048
    elif model_type == "ResNet34_3D_MAE":
        model = ResNet34_3D_MAE(in_chans=1)
        embed_dim = 512
    else:
        raise ValueError(f"Modelo {model_type} no reconocido.")
        
    # 3. CARGA DE PESOS CONTROLADA (La clave del Baseline)
    if pretrained and checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f" Cargando pesos preentrenados por Curriculum desde: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            
            # Limpieza defensiva de prefijos inspirada en el código de BrainIAC
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    new_state_dict[k[4:]] = v
                elif k.startswith("backbone."):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f" Alerta: Checkpoint no encontrado en {checkpoint_path}. Usando pesos aleatorios.")
    else:
        print(" Pesos inicializados ALEATORIAMENTE (Baseline sin preentrenamiento).")
        
    model.to(device)
    return model, embed_dim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# 2. Function to filter a balanced few-shot dataset (K examples per class)
def get_few_shot_indices(dataset, K, num_classes):
    class_counts = {c: 0 for c in range(num_classes)}
    few_shot_indices = []
    
    # Iterate over the training dataset indices
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if class_counts[label] < K:
            few_shot_indices.append(idx)
            class_counts[label] += 1
        if all(count == K for count in class_counts.values()):
            break
    return few_shot_indices
class FineTuneWrapper(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, is_regression=False):
        super().__init__()
        self.backbone = backbone  # Tu modelo ResNet_3D_MAE
        self.is_regression = is_regression
        
        # Cabecera lineal de adaptación (Linear Head)
        if is_regression:
            self.classifier = nn.Linear(embed_dim, 1)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)
            
    def forward(self, x):
        # Desempaquetamos la tupla del MAE (omitimos la reconstrucción volumétrica)
        _, z_global = self.backbone(x) 
        
        # z_global ya viene con la media reducida del volumen dim=(2,3,4) en tu curriculum
        logits = self.classifier(z_global)
        
        return logits.squeeze(-1) if self.is_regression else logits
    

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_absolute_error, r2_score

def run_pytorch_finetuning(model_type, checkpoint_path, train_dataset, val_dataset, task_config, pretrained=True, epochs=15, device="cuda"):
    """
    Ejecuta el fine-tuning completo actualizando pesos y evaluando métricas en cada época.
    """
    # 1. Cargar el backbone base (Preentrenado o Aleatorio)
    backbone, embed_dim = get_model(model_type, checkpoint_path, device, pretrained=pretrained)
    
    # 2. Configurar parámetros de la tarea downstream
    task_type = task_config["task_type"]
    num_classes = task_config["num_classes"]
    
    is_regression = (task_type == "regression")
    
    # 3. Envolver en el clasificador adaptado
    model = FineTuneWrapper(backbone, embed_dim, num_classes, is_regression=is_regression).to(device)
    
    # Asegurar que TODAS las capas participen del cálculo de gradientes
    model.train() 
    for param in model.parameters():
        param.requires_grad = True
        
    # 4. Optimizador con Learning Rates diferenciales (Estrategia del Paper)
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},    # LR muy bajo para el foundation model
        {'params': model.classifier.parameters(), 'lr': 1e-3}   # LR normal para la cabecera lineal
    ], weight_decay=1e-4)
    
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Extraemos el sufijo (ej. "_25pc"). Si no hay, queda vacío "".
    sufijo = task_config.get("suffix", "")
    log_file_name = f"scratch_partitions_trials_please_bestmodel/results_scarcity_bestmodel/finetuning_metrics_scratch{sufijo}.txt"

    directorio_log = os.path.dirname(log_file_name)
    if directorio_log:
        os.makedirs(directorio_log, exist_ok=True)
    
    # Creamos/Limpiamos el archivo al inicio del entrenamiento y escribimos la cabecera
    with open(log_file_name, "w") as f_log:
        f_log.write(f"Iniciando Entrenamiento - Epochs: {epochs}\n")
        f_log.write("Epoch | Loss Train | Loss Val | Metricas\n")
        f_log.write("-" * 50 + "\n")


    print(f"\n Iniciando Actualización de Pesos ({'Preentrenado' if pretrained else 'Desde Cero'}). Épocas: {epochs}")
    print(f" Guardando logs en: {log_file_name}")


    print(f" Guardando logs en: {log_file_name}")
    best_auc = 0.0  # AÑADIR ESTO: Rastreador del mejor AUC

    best_val_metric = -float('inf') # Rastreará la mejor métrica (R2 o AUC)
    best_model_wts = copy.deepcopy(model.state_dict()) # Guarda los pesos iniciales por si acaso

    for epoch in range(epochs):
        # --------------------------------------------------
        # FASE A: ENTRENAMIENTO (ACTUALIZACIÓN DE PESOS)
        # --------------------------------------------------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            label_idx = task_config["label_idx"]
            labels = batch[label_idx]
            images = batch[0].to(device)
            labels = batch[label_idx].to(device) 
            labels = labels.float() if is_regression else labels.long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --------------------------------------------------
        # FASE B: VALIDACIÓN Y MÉTRICAS (¡FALTABA ESTO!)
        # --------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[label_idx].to(device)
                #label_idx = task_config["label_idx"]
                #labels = batch[label_idx]
                labels = labels.float() if is_regression else labels.long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                if is_regression:
                    all_preds.extend(outputs.cpu().numpy())
                else:
                    # Guardamos probabilidades softmax de la clase positiva para calcular AUC
                    probs = F.softmax(outputs, dim=1)
                    if num_classes == 2:
                        all_preds.extend(probs[:, 1].cpu().numpy()) #Probabilidad clase 1
                    else:
                        all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 5. Cálculo Dinámico de Métricas según la Tarea
        metrics_str = ""
        current_val_metric = 0.0 # Variable para decidir si es el mejor modelo

        if is_regression:
            r2 = r2_score(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            metrics_str = f"R2: {r2:.4f} | MAE: {mae:.4f}"
            current_val_metric = r2 # En regresión, maximizamos R2
        else:
            if num_classes == 2:
                auc_score = roc_auc_score(all_labels, all_preds)
                preds_bin = (all_preds >= 0.5).astype(int)
                bal_acc = balanced_accuracy_score(all_labels, preds_bin)
                metrics_str = f"AUC: {auc_score:.4f} | BalAcc: {bal_acc:.4f}"
                current_val_metric = auc_score # En binaria, maximizamos AUC
            else:
                # Para multiclase, asumo que quieres maximizar Balanced Accuracy o podrías calcular AUC multi-clase
                preds_bin = np.argmax(all_preds, axis=1)
                bal_acc = balanced_accuracy_score(all_labels, preds_bin)
                metrics_str = f"BalAcc: {bal_acc:.4f}"
                current_val_metric = bal_acc

        print(f" 🗓️ Época {epoch+1}/{epochs} -> Loss Train: {avg_train_loss:.4f} | Loss Val: {avg_val_loss:.4f} | {metrics_str}")
        
        # GUARDAR LOS MEJORES PESOS 
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict()) # Clonamos los pesos ganadores
            print(f"  ¡Nuevo mejor modelo! Métrica: {best_val_metric:.4f}")

      
        # Abrimos en modo "a" (append) para añadir una nueva línea sin borrar lo anterior
        with open(log_file_name, "a") as f_log:
            f_log.write(f"{epoch+1} | {avg_train_loss:.4f} | {avg_val_loss:.4f} | {metrics_str}\n")

    # 6. GUARDAR CHECKPOINT FINAL DE ESTE EXPERIMENTO (¡FALTABA ESTO!)
    #mode_name = "pretrained_curriculum" if pretrained else "random_scratch"
    #output_path = f"finetuned_{model_type}_{mode_name}.pth"
    # --- NUEVO: RESTAURAR EL MEJOR MODELO ANTES DE GUARDAR Y SALIR ---
    print(f"\nEntrenamiento completado. Cargando los pesos de la mejor época (Métrica Val: {best_val_metric:.4f})...")
    model.load_state_dict(best_model_wts)

    # 6. GUARDAR CHECKPOINT FINAL DE ESTE EXPERIMENTO
    sufijo = task_config.get("suffix", "")
    checkpoint_name = f"scratch_partitions_trials_please_bestmodel/results_scarcity_bestmodel/best_downstream_model_scratch{sufijo}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'task_config': task_config,
        'model_type': model_type
    }, checkpoint_name)
    print(f"Mejor modelo guardado con éxito en: {checkpoint_name}")
        
    return model, best_val_metric


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


# ======================================================
# 3. EXTRACCIÓN Y CACHÉ DE EMBEDDINGS
# ======================================================
# Añade 'embeddings_dir' como último parámetro
def get_or_extract_embeddings(epoch, dataloader, model_type, checkpoint_path, embeddings_dir):
    cache_file = f"{embeddings_dir}/features_epoch_{epoch}_v4_amyloid_mci_age_gender.pkl" 
    
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    
    print(f"Extrayendo características para epoch {epoch}...")
    model, _ = get_model(model_type, checkpoint_path, DEVICE)
    
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
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, r2_score, mean_absolute_error


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
    
    # SOLUCIÓN CRÍTICA 3: Filtrar para quedarse solo con las medias de AUROC o la tarea multiclase, evitando barras de STD inservibles
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

def evaluar_en_test(model, test_loader, device, task_config):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # 1. CORRECCIÓN DEL AMARILLO: Extraemos las variables desde el diccionario de configuración
    is_regression = task_config.get("task_type") == "regression"
    num_classes = task_config.get("num_classes", 2) 

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            label_idx = task_config["label_idx"]

            labels = batch[label_idx]
            
            if images.dim() == 4:
                images = images.unsqueeze(0)  # Convierte [C, D, H, W] en [1, C, D, H, W]

            outputs = model(images)
            
            if is_regression:
                preds = outputs.squeeze(-1).cpu().numpy()
                all_preds.extend(preds)
            else:
                # Calculamos las probabilidades con Softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # 2. ADAPTACIÓN DE DIMENSIONES SEGÚN EL TIPO DE TAREA
                if num_classes == 2:
                    # Binario: Guardamos solo la probabilidad de la clase positiva (Array 1D)
                    preds = probs[:, 1]
                    all_preds.extend(preds)
                else:
                    # Multiclase: Guardamos la matriz completa de probabilidades (Array 2D)
                    preds = probs
                    all_preds.extend(preds)

            if isinstance(labels, torch.Tensor):
                if labels.dim() == 0:
                    all_targets.append(labels.item())
                else:
                    all_targets.extend(labels.cpu().numpy())
            else:
                all_targets.append(labels)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. DIAGNÓSTICO PARA EL PROBLEMA DE sMCI vs pMCI (Evitar NaNs silenciosos)
    if not is_regression:
        # Alerta A: El modelo ha colapsado y predice valores inválidos (Explosión de gradientes)
        if np.isnan(all_preds).any():
            print("\n ¡ALERTA! Las predicciones del modelo contienen valores NaN. El entrenamiento ha colapsado.")
            return float('nan')
        
        # Alerta B: No hay suficientes clases en el conjunto de test debido a la escasez extrema
        clases_unicas = np.unique(all_targets)
        if len(clases_unicas) < 2:
            print(f"\n ¡ALERTA! El conjunto de TEST solo contiene la clase {clases_unicas}. El AUC no se puede calcular.")
            return float('nan')

    # 4. CÁLCULO DE MÉTRICAS (Línea duplicada incorrecta eliminada)
    if is_regression:
        metric_value = r2_score(all_targets, all_preds)
    else:
        if num_classes == 2:
            metric_value = roc_auc_score(all_targets, all_preds) 
        else:
            # Para tareas multiclase reales (ej. DX con 3 clases: CN, MCI, AD)
            metric_value = roc_auc_score(all_targets, all_preds, multi_class='ovo')
        
    return metric_value
       

# ======================================================
# PREPARACIÓN DE DATOS Y COLA DE EXPERIMENTOS
# ======================================================
import argparse
if __name__ == "__main__":
    import argparse
    import traceback
    import torch
    import pandas as pd
    import nibabel as nib
    import re
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    # CONFIGURACIÓN DE ENTORNO (0: Local, 1: Clúster)
    USE_CLUSTER = 0 

    # --- PASO 0: GESTIÓN DE ARGUMENTOS Y DISPOSITIVO ---
    if USE_CLUSTER == 1:
        parser = argparse.ArgumentParser(description="Lanza el análisis downstream de un modelo específico.")
        parser.add_argument("--config", type=int, required=True, help="ID del experimento a ejecutar (índice de la lista)")
        args = parser.parse_args()
        config_id = args.config

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            print(f"PASO 0 (Clúster): Verificando GPU... (Capacidad CUDA: {major}.{minor})")
            if major < 7:
                raise RuntimeError(f"❌ ERROR: La GPU asignada es demasiado antigua ({major}.{minor}). Se requiere >= 7.0.")
        else:
            print("PASO 0 (Clúster): No se detectó GPU. Ejecutando en CPU.")
    else:
        config_id = 0  # En modo local, ejecutamos por defecto el primer experimento (índice 0)
        print(" PASO 0 (Local): Ejecutando en modo manual local.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Usando dispositivo: {DEVICE}")

    # --- PARÁMETROS DE RUTAS Y CONFIGURACIÓN ---
    ROOT_DIR = "/export/data_ml4ds/Neurocosas/databases/ADNI/M00"
    CSV_PATH = "/export/usuarios01/nbesteban/ADNIMERGE_25Aug2023.csv"
    AMYLOID_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/UCBERKELEYAV45_amyloid_status.xlsx" 
    MCI_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/Classification_MCI_to_AD.xlsx"
    PHC_XLSX_PATH = "/export/data_ml4ds/Neurocosas/FOUNDATION_MODELS/NEW/curriculum_patches/analysis/code/downstream/annual_memory_change_phc.xlsx"
    
    COMPATIBLE_SIZE = (128, 160, 128)
    BATCH_SIZE = 4

    try:
        # --- PASO 1: CARGA DE EXCEL Y CSV DE ETIQUETAS ---
        print(" PASO 1: Cargando excels y CSVs de etiquetas...")
        amyloid_df = pd.read_excel(AMYLOID_XLSX_PATH)
        rid_to_amyloid = dict(zip(amyloid_df["RID"].astype(int), amyloid_df["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"]))

        mci_df = pd.read_excel(MCI_XLSX_PATH)
        mci_map = {"Stable-MCI": 0, "Converter-MCI": 1}
        rid_to_mci = {int(row["RID"]): mci_map.get(row["CLASSIFICATION"], -1) 
                      for _, row in mci_df.dropna(subset=["RID"]).iterrows()}




        demographics_df = pd.read_csv(CSV_PATH, low_memory=False) 





        rid_to_age = dict(zip(demographics_df['RID'], demographics_df['AGE']))
        rid_to_gender = dict(zip(demographics_df['RID'], demographics_df['PTGENDER']))
        gender_map = {"Male": 0, "Female": 1}

        phc_df = pd.read_excel(PHC_XLSX_PATH).dropna(subset=["RID", "PHC_mem_rate"]) 
        rid_to_phc = dict(zip(phc_df["RID"].astype(int), phc_df["PHC_mem_rate"]))
        
        # --- EN TU PASO 1 ---
        labels_df = demographics_df.copy()
        labels_df["PTID"] = labels_df["PTID"].str.strip()
        
        # 1. Filtrar el dataframe para quedarnos SOLAMENTE con AD y CN
        #labels_df = labels_df[labels_df["DX_bl"].isin(["CN", "AD"])].copy()
        #  AÑADIR ESTO: Configuración global de tareas
        
        id_to_data = {row["PTID"]: row["DX_bl"] for _, row in labels_df.iterrows()}
        
        diag_to_label = {"CN": 0, "SMC": 1, "AD": 2, "LMCI": 4, "EMCI": 5}

        # --- EN TU PASO 2 ---
        print(" PASO 2: Buscando y cruzando archivos NIfTI...")
        all_nii = find_nii_files(ROOT_DIR)
        nii_info = []
        
        for path in all_nii:
            subj_id = get_subject_id(path) 
            if subj_id in id_to_data:
                dx_raw = id_to_data[subj_id]
                dx_label = diag_to_label.get(dx_raw, -1)

                # 3. Ignorar explícitamente los sujetos que no son AD ni CN
                #if dx_label == -1:
                 #   continue

                match = re.search(r'(\d{3})S(\d{4})', str(path))
                rid = int(match.groups()[1]) if match else -1
                
                phc_label = rid_to_phc.get(rid, -1000.0)
                amyloid_label = rid_to_amyloid.get(rid, -1)
                mci_label = rid_to_mci.get(rid, -1)
                age_label = rid_to_age.get(rid, -1.0)
                gender_label = gender_map.get(rid_to_gender.get(rid, "Unknown"), -1)

                nii_info.append((rid, path, dx_label, phc_label, amyloid_label, mci_label, age_label, gender_label))

        # --- PASO 3: VALIDACIÓN DE INTEGRIDAD DEL NIFTI ---
        print(" PASO 3: Validando integridad física de los NIfTIs...")
        valid_rows = []
        for row in tqdm(nii_info):
            path = row[1]
            try:
                nib.load(path)
                valid_rows.append(row)
            except Exception:
                continue

        nii_df = pd.DataFrame(valid_rows, columns=[
            "rid", 
            "path", 
            "label_dx", 
            "label_phc", 
            "label_amyloid", 
            "label_mci", 
            "label_age", 
            "label_gender"
        ])
        print(f" Dataframe consolidado con {len(nii_df)} sujetos válidos.")

        
        # --- PASO 5: MAPPING DE EXPERIMENTOS ---
        # Lista centralizada de configuraciones. El script tomará la del índice asignado por config_id
        lista_experimentos = [
            {"name": "2_resnet_MAE_L_checkpoints_loss-ssim_tissue-True_noBG-True_m-0.75_patch-16_symTrue", 
             "type": "ResNet50_3D_MAE", 
             "checkpoint": "tu_ruta_al_checkpoint_de_curriculum.pth"},
            
        ]

        if config_id >= len(lista_experimentos):
            print(f" Alerta: El config ID {config_id} no está en la lista. Revertiendo al índice 0.")
            config_id = 0

        exp_actual = lista_experimentos[config_id]
        print(f" Configuración Activa -> Modelo: {exp_actual['type']} | Checkpoint: {exp_actual['name']}")

        # --- PASO 6: CONFIGURACIÓN DE LA TAREA (Estilo BrainIAC) ---
        TASKS_CONFIG = {
            "diagnostico_alzheimer": {"task_type": "classification", "num_classes": 2, "label_idx": 1},
            #"edad_cerebral":         {"task_type": "regression",     "num_classes": 1, "label_idx": 5}
        }
        TASKS_CONFIG = {
            "AD_vs_CN": {
                "task_type": "classification",
                "num_classes": 2,
                "label_idx": 1
            },
            "sMCI_vs_pMCI": {
                "task_type": "classification",
                "num_classes": 2,
                "label_idx": 4
            },
            "Gender": {
                "task_type": "classification",
                "num_classes": 2,
                "label_idx": 6
            },
            #for multiclass
            "AD_vs_MCI_vs_CN": {
                "task_type": "classification",
                "num_classes": 3,
                "label_idx": 1
            },

            #{"name": "PHC", "task_type": "regression", "num_classes": 1, "label_idx": 3},
            #{"name": "Age", "task_type": "regression", "num_classes": 1, "label_idx": 6},

        }
        

        
        fracciones_log = np.logspace(np.log10(0.01), np.log10(0.25), num=4)
        
        # 2. Definimos los porcentajes fijos del resto del experimento
        fracciones_resto = np.array([0.50, 0.75, 1.00])
        
        # 3. Concatenamos ambas listas y nos aseguramos de que sean valores únicos ordenados
        fracciones = np.unique(np.concatenate([fracciones_log, fracciones_resto]))

        SEMILLAS_A_PROCESAR = [42, 100, 134, 4332, 2026, 999, 29, 344, 283, 22]
        # Cambiamos tu lista por un diccionario para extraer el valor real del porcentaje
        FRACCIONES_DICT = {"1pc": 0.01, "2pc": 0.02, "8pc": 0.08, "25pc": 0.25, "50pc": 0.50, "75pc": 0.75, "100pc": 1.0}
        TOTAL_EPOCHS = 50 
        DIRECTORIO_SALIDA = "./scratch_partitions_trials_please_bestmodel/resultados_scarcity_finetuning_seeds_tasks" 
        os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
        
        TASKS = [
            {"name": "AD_vs_CN", "task_type": "classification", "num_classes": 2, "label_idx": 1},
            #{"name": "PHC", "task_type": "regression", "num_classes": 1, "label_idx": 2},
            {"name": "Amyloid", "task_type": "classification", "num_classes": 2, "label_idx": 3},
            {"name": "sMCI_vs_pMCI", "task_type": "classification", "num_classes": 2, "label_idx": 4}, 
            #{"name": "Age", "task_type": "regression", "num_classes": 1, "label_idx": 5},
            {"name": "Gender", "task_type": "classification", "num_classes": 2, "label_idx": 6}
        ]

        for seed in SEMILLAS_A_PROCESAR:
            print(f"\n=========================================")
            print(f" INICIANDO EXPERIMENTOS PARA SEMILLA: {seed}")
            print(f"=========================================")
            
            # Inicializamos el generador de números aleatorios para esta corrida
            set_all_seeds(seed)
            
            # Diccionario para almacenar TODAS las métricas de esta semilla (Múltiples Tareas y Fracciones)
            metricas_de_esta_semilla = {}
            
            # NUEVO: BUCLE DE TAREAS 
            for task_config in TASKS:
                task_name = task_config["name"]
                print(f"\n Evaluando tarea: {task_name}")
                
                # --- 1. FILTRADO ESPECÍFICO DE LA TAREA ---
                # Partimos del dataframe completo (asumiendo que se llama nii_df)
                df_task = nii_df.copy()
                
                if task_name == "AD_vs_CN":
                    
                    df_task = df_task[df_task["label_dx"].isin([0, 1, 2])].copy()

                    df_task["label_dx"] = np.where(
                        df_task["label_dx"] == 2,
                        1,
                        0
                    ).astype(int)

                    col_stratify = "label_dx"

                elif task_name == "sMCI_vs_pMCI":
                    # Nos quedamos estrictamente con 0.0 y 1.0 (sMCI y pMCI reales)
                    df_task = df_task[df_task["label_mci"].isin([0, 1])].copy()
                    # Comprobación de seguridad
                    if len(df_task) == 0:
                        raise ValueError("Error: El DataFrame de MCI está vacío.")
                        
                    df_task["label_mci"] = df_task["label_mci"].astype(int)
                    col_stratify = "label_mci"

                elif task_name == "Amyloid":
                    df_task = df_task[df_task["label_amyloid"] != -1]
                    col_stratify = "label_amyloid"
                elif task_name == "PHC":
                    df_task = df_task[(df_task["label_phc"] != -1000.0) & (df_task["label_phc"].notna())]
                    col_stratify = None # No se estratifica regresión
                elif task_name == "Age":
                    df_task = df_task[df_task["label_age"] > 0]
                    col_stratify = None # No se estratifica regresión
                elif task_name == "Gender":
                    df_task = df_task[df_task["label_gender"] != -1]
                    col_stratify = "label_gender"

                
                # --- 3. BUCLE DE FRACCIONES TOTALMENTE ARREGLADO Y ADAPTADO ---
                for porcentaje_str, frac in FRACCIONES_DICT.items():
                    print(f"\n Alineando particiones para Semilla: {seed} | Tarea: {task_name} | Fracción: {porcentaje_str}")
            
                    # 1. CONSTRUIR LA RUTA AL CSV DE LA PARTICIÓN CORRESPONDIENTE
                    ruta_csv_split = f"rids_splits_cache/split_seed_{seed}_task_{task_name}_frac_{porcentaje_str}.csv"
                    
                    if not os.path.exists(ruta_csv_split):
                        raise FileNotFoundError(f" No se encontró el split guardado en: {ruta_csv_split}. Ejecuta primero el código de embeddings.")
                    
                    # 2. LEER EL ARCHIVO DE RIDs ALINEADO
                    df_split_actual = pd.read_csv(ruta_csv_split)
                    
                    # Separar los RIDs según el conjunto asignado en el archivo original
                    rids_train = df_split_actual[df_split_actual['Set'] == 'Train']['RID'].values
                    rids_val   = df_split_actual[df_split_actual['Set'] == 'Validation']['RID'].values
                    rids_test  = df_split_actual[df_split_actual['Set'] == 'Test']['RID'].values
                    
                    # 3. FILTRAR TU DATAFRAME REAL UTILIZANDO LOS RIDs EXACTOS (Usa df_task en minúsculas 'rid')
                    df_train_task = df_task[df_task['rid'].isin(rids_train)].copy()
                    df_val_task   = df_task[df_task['rid'].isin(rids_val)].copy()
                    df_test_task  = df_task[df_task['rid'].isin(rids_test)].copy()
                    
                    print(f" Muestras activas de Train para este ciclo ({porcentaje_str}): {len(df_train_task)}")

                    # 4. INSTANCIAR LOS DATASETS NATIVOS CON TUS CONFIGURACIONES REALES
                    inference_dataset_train_sub = InferenceNiiDataset(df_train_task, target_size=COMPATIBLE_SIZE)
                    inference_dataset_val       = InferenceNiiDataset(df_val_task, target_size=COMPATIBLE_SIZE)
                    inference_dataset_test      = InferenceNiiDataset(df_test_task, target_size=COMPATIBLE_SIZE)

                    # Configuramos el sufijo para que los .pth y .txt se guarden con nombres únicos
                    config_actual = task_config.copy()
                    config_actual["suffix"] = f"_{task_name}_{porcentaje_str}_seed_{seed}"

                    # 5. EJECUTAMOS EL ENTRENAMIENTO (Pasando los datasets alineados por RID)
                    modelo_scratch, mejor_auroc_val = run_pytorch_finetuning(
                        model_type=exp_actual["type"], 
                        checkpoint_path=None, 
                        train_dataset=inference_dataset_train_sub,
                        val_dataset=inference_dataset_val,
                        task_config=config_actual,
                        pretrained=False, 
                        epochs=TOTAL_EPOCHS,
                        device=DEVICE
                    )

                    print("🔬 Evaluando el modelo final en el conjunto de TEST...")
                    test_metric = evaluar_en_test(modelo_scratch, inference_dataset_test, DEVICE, config_actual)

                    # Extraemos el número para el json
                    pct_numero = int(porcentaje_str.replace("pc", ""))

                    # --- 4. GUARDADO DINÁMICO EN EL DICCIONARIO ---
                    metric_name = "R2" if config_actual["task_type"] == "regression" else "auroc"
                    clave_metrica = f"{pct_numero}%_{task_name}_{metric_name}"
                    
                    metricas_de_esta_semilla[clave_metrica] = test_metric
                    
                    print(f" Tarea {task_name} | Fracción {porcentaje_str} completada. Test {metric_name.upper()}: {test_metric:.4f} | .pth guardado.")
            # --- 5. AL GUARDAR EL JSON, AHORA CONTIENE TODAS LAS TAREAS ---
            ruta_json_final = os.path.join(DIRECTORIO_SALIDA, f"metricas_epoch_{TOTAL_EPOCHS}_seed_{seed}.json")
            with open(ruta_json_final, 'w') as f:
                json.dump(metricas_de_esta_semilla, f, indent=4)
            print(f"Archivo de métricas (Multi-Tarea) guardado exitosamente: {ruta_json_final}")

    except Exception as e:
        print("\n" + "x"*20)
        print("EL SCRIPT HA EXPIRADO O CRASHEADO. DETALLES DEL ERROR:")
        print(traceback.format_exc())
        print("x"*20 + "\n")

