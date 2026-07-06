import os
import re
import argparse
import traceback
from pathlib import Path
from datetime import datetime
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
 
import wandb
from monai.losses import SSIMLoss
from pytorch_msssim import ms_ssim
from fused_ssim import FusedSSIMMap

#Global Constants
EXPERIMENT_NAME = "final_comparison_loss"
 
COMPATIBLE_SIZE = (128, 160, 128)
ORIGINAL_SIZE   = (113, 137, 113)
BATCH_SIZE      = 2
EFFECTIVE_BATCH_SIZE = 4
ACCUMULATION_STEPS   = max(1, EFFECTIVE_BATCH_SIZE // BATCH_SIZE)
NUM_WORKERS     = 4
 
DATA_FOLDERS = [
    "/export/data_ml4ds/Neurocosas/databases/DLBS",
    "/export/data_ml4ds/Neurocosas/databases/IXI",
    "/export/data_ml4ds/Neurocosas/databases/SALD",
    "/export/data_ml4ds/Neurocosas/databases/Wayne/EEF",
    "/export/data_ml4ds/Neurocosas/databases/Wayne/10",
    "/export/data_ml4ds/Neurocosas/databases/Wayne/11",
    "/export/data_ml4ds/Neurocosas/databases/uc3m_nki_cat1290_mwp1",
    "/export/data_ml4ds/Neurocosas/databases/CamCan2",
    "/export/data_ml4ds/Neurocosas/databases/OASIS_Cleaned_Niftis",
]
 
 
# Dataset
class TensorDataset(Dataset):
    """Carga tensores preprocesados (.pt) desde disco."""
 
    def __init__(self, tensor_paths, compatible_size=(128, 160, 128)):
        self.tensor_paths    = tensor_paths
        self.compatible_size = compatible_size
 
    def __len__(self):
        return len(self.tensor_paths)
 
    def __getitem__(self, idx):
        path = self.tensor_paths[idx]
        try:
            return torch.load(path, map_location="cpu").clone()
        except Exception as e:
            print(f"[ERROR] Al cargar el tensor {path}: {e}")
            return torch.zeros((1, *self.compatible_size))
 
 

# MODEL: ResNet-50 3D AUTOENCODER
class BasicBlock3D(nn.Module):
    """Bloque residual estándar de ResNet-18/34."""
 
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )
 
    def forward(self, x):
        out  = self.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)
 
class Bottleneck3D(nn.Module):
    """Bloque residual Bottleneck para arquitecturas ResNet grandes (50/101/152)."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(self.expansion * planes)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet50_3D_MAE(nn.Module):
    """
    Autoencoder 3D basado en ResNet-50 (ResNet-L).
    Devuelve (reconstrucción, vector latente z).
    """

    def __init__(self, in_chans=1):
        super().__init__()

        # encoder
        self.in_planes = 64
        self.conv1   = nn.Conv3d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 configuration: [3, 4, 6, 3] bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck3D, 64,  3, stride=1)  # output with 256 ch
        self.layer2 = self._make_layer(Bottleneck3D, 128, 4, stride=2)  # output with 512 ch
        self.layer3 = self._make_layer(Bottleneck3D, 256, 6, stride=2)  # output with 1024 ch
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)  # output with 2048 ch

        # decoder
        self.dec4      = self._dec_block(2048, 512)
        self.dec3      = self._dec_block(512, 256)
        self.dec2      = self._dec_block(256, 128)
        self.dec1_pool = self._dec_block(128, 64)
        self.dec0_conv1 = self._dec_block(64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv3d(64, in_chans, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _dec_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e0     = self.relu(self.bn1(self.conv1(x)))
        e_pool = self.maxpool(e0)
        e1 = self.layer1(e_pool)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        z  = self.layer4(e3)  # dimensions [B, 2048, 4, 5, 4]

        # Decoder
        d = self.dec4(z)
        d = self.dec3(d)
        d = self.dec2(d)
        d = self.dec1_pool(d)
        d = self.dec0_conv1(d)

        recon = self.final_conv(d)
        return recon, z.mean(dim=(2, 3, 4))

 
 
 
# for the loss function
class SSIM3DMapLoss(nn.Module):
    """
    Calcula el SSIM en 3D y devuelve un mapa del mismo tamaño que la imagen.
    Permite multiplicar el resultado por la máscara del MAE.
    """
 
    def __init__(self, window_size=5, val_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        self.C1  = (0.01 * val_range) ** 2
        self.C2  = (0.03 * val_range) ** 2
 
    def forward(self, pred, target):
        pool = nn.AvgPool3d(kernel_size=self.window_size, stride=1, padding=self.pad)
 
        mu1, mu2   = pool(pred), pool(target)
        mu1_sq     = mu1 ** 2
        mu2_sq     = mu2 ** 2
        mu1_mu2    = mu1 * mu2
 
        sigma1_sq = pool(pred ** 2)    - mu1_sq
        sigma2_sq = pool(target ** 2)  - mu2_sq
        sigma12   = pool(pred * target) - mu1_mu2
 
        cs_map   = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = ((2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)) * cs_map
 
        return 1.0 - ssim_map
 
 
# Curriculum learning configuration
def get_curriculum_params(epoch):
    """
    Estrategia progresiva:
    1. Fase Calentamiento (0-100): Aprender texturas locales. Mask baja, Patch pequeño.
    2. Fase Consolidación (100-400): Aumentar dificultad de reconstrucción. Mask media.
    3. Fase Estructural (400-700): Aprender formas globales. Patch mediano.
    4. Fase Experta (700+): Reconstrucción semántica difícil. Patch grande, Mask alta.
    
    Nota: Los tamaños de patch deben ser divisores de (120, 160, 120).
    Divisores comunes seguros: 5, 10, 20, 40.
    """
    # Phase 1
    if epoch < 100:
        patch_size = 16    # small
        mask_ratio = 0.30  # low difficulty

    # Phase 2
    elif epoch < 200:
        patch_size = 16    
        mask_ratio = 0.50  

    # Phase 3
    elif epoch < 300:
        patch_size = 16   
        mask_ratio = 0.75 

    # Phase 4...
    elif epoch < 400:
        patch_size = 32    
        mask_ratio = 0.3  

    elif epoch < 500:
        patch_size = 32    
        mask_ratio = 0.5

    elif epoch < 600:
        patch_size = 32    
        mask_ratio = 0.75

    else:
        patch_size = 32
        mask_ratio = 0.75

    return mask_ratio, patch_size
 

class FusedMSSSIM3DMapLoss(nn.Module):
    """
    Calculates Multi-Scale SSIM in 3D using the ultra-efficient kernels from Fused-SSIM.
    Maintains the parallelization by scales using CUDA streams and
    returns a map that is compatible with the masks from MAE.
    """
    def __init__(self, weights=None, padding="same"):
        super().__init__()
        self.padding = padding
        # Standard C1 and C2 constants that their repo uses
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        # standard weights for 4 scales in 3D volumes
        if weights is None:
            self.weights = [0.0448, 0.2856, 0.3001, 0.3695]
        else:
            self.weights = weights
        self.num_scales = len(self.weights)

    def forward(self, pred, target):
        # ensuring that the tesnors comply with the CUDA requisite
        pred   = pred.contiguous().float()
        target = target.contiguous().float()
        
        B, C, D, H, W = pred.shape
        device = pred.device
        
        preds   = [pred]
        targets = [target]

        # downsampling 
        for i in range(self.num_scales - 1):
            preds.append(F.avg_pool3d(preds[-1], kernel_size=2, stride=2))
            targets.append(F.avg_pool3d(targets[-1], kernel_size=2, stride=2))

        # parallel calculation of each scale using CUDA Streams
        streams = [torch.cuda.Stream(device=device) for _ in range(self.num_scales)]
        results = [None] * self.num_scales

        for i in range(self.num_scales):
            with torch.cuda.stream(streams[i]):
                # Autograd to obtain the spatial map, not mean scalar
                # Important: preds[i] goes first because it requires gradients; targets[i] doesn't.
                ssim_map = FusedSSIMMap.apply(
                    self.C1, self.C2, preds[i], targets[i], 
                    self.padding, self.training, 3  # 3 is spatial_dims=3
                )
                
                # Interpolate the returned map to the original size of the mask (128x160x128)
                if i > 0:
                    ssim_map = F.interpolate(ssim_map, size=(D, H, W), mode='trilinear', align_corners=False)
                
                results[i] = ssim_map

        # synchronizing the GPU flows
        torch.cuda.synchronize(device=device)

        # combination of the scales using the structural exponents
        msssim_map = torch.ones_like(pred)
        weights_tensor = torch.tensor(self.weights, device=device)

        for i in range(self.num_scales):
            weight = weights_tensor[i]
            # Clamping with ReLU to avoid mathematical problems if a pixel is negative before applying the expornent
            msssim_map *= (F.relu(results[i]) ** weight)

        # to return 1 - MS-SSIM
        return 1.0 - msssim_map
     
# Masking patches
def mask_patches(x, patch_size, mask_ratio, force_tissue_ratio=False, symmetric=False, shape_type="random"):
    B, C, D, H, W = x.shape
 
    # size of the patch grid
    grid_D = (D + patch_size - 1) // patch_size
    grid_H = (H + patch_size - 1) // patch_size
    grid_W = (W + patch_size - 1) // patch_size
 
    # Padding so that the volumen is common multiple of the patch
    pad_d = grid_D * patch_size - D
    pad_h = grid_H * patch_size - H
    pad_w = grid_W * patch_size - W
    x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), "constant", 0)
 
    # tissue ratio per patch
    if force_tissue_ratio:
        tissue_mask  = (x_padded > 0).float()
        tissue_ratio = F.avg_pool3d(tissue_mask, kernel_size=patch_size, stride=patch_size)
        tissue_ratio = tissue_ratio.squeeze(1)   # [B, grid_D, grid_H, grid_W]
    else:
        tissue_ratio = None
 
    if symmetric:
        if grid_D % 2 != 0:
            grid_D += 1
            if tissue_ratio is not None:
                tissue_ratio = F.pad(tissue_ratio, (0, 0, 0, 0, 0, 1), "constant", 0)
 
        total_voxels_D  = grid_D * patch_size
        pad_left        = (total_voxels_D - D) // 2
        half_grid_D     = grid_D // 2
        num_patches_half = half_grid_D * grid_H * grid_W
        num_mask_half    = int(mask_ratio * num_patches_half)
    else:
        num_patches_total = grid_D * grid_H * grid_W
        num_mask          = int(mask_ratio * num_patches_total)
 
    # mask construction per batch sample
    final_masks = []
    for b in range(B):
        if symmetric:
            m_half = torch.zeros(half_grid_D, grid_H, grid_W, device=x.device)
            if force_tissue_ratio and tissue_ratio is not None:
                t_half        = tissue_ratio[b, :half_grid_D, :, :]
                valid_indices = torch.where(t_half.view(-1) >= 0.20)[0]
                if len(valid_indices) >= num_mask_half:
                    perm = torch.randperm(len(valid_indices), device=x.device)[:num_mask_half]
                    idx  = valid_indices[perm]
                else:
                    idx = torch.randperm(num_patches_half, device=x.device)[:num_mask_half]
            else:
                idx = torch.randperm(num_patches_half, device=x.device)[:num_mask_half]
 
            m_half.view(-1)[idx] = 1
            m_flipped = torch.flip(m_half, dims=[0])
            m_full    = torch.cat([m_half, m_flipped], dim=0)
            final_masks.append(m_full)
        else:
            m = torch.zeros(grid_D, grid_H, grid_W, device=x.device)
            if force_tissue_ratio and tissue_ratio is not None:
                valid_indices = torch.where(tissue_ratio[b].view(-1) >= 0.20)[0]
                if len(valid_indices) >= num_mask:
                    perm = torch.randperm(len(valid_indices), device=x.device)[:num_mask]
                    idx  = valid_indices[perm]
                else:
                    idx = torch.randperm(m.numel(), device=x.device)[:num_mask]
            else:
                idx = torch.randperm(m.numel(), device=x.device)[:num_mask]
            m.view(-1)[idx] = 1
            final_masks.append(m)
 
    # expanding patches to voxels
    patch_mask = torch.stack(final_masks)
    voxel_mask_base = (
        patch_mask
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
        .repeat_interleave(patch_size, dim=3)
    )
 
    # Creating the geometrical template
    coords = torch.arange(patch_size, dtype=torch.float32, device=x.device)
    center = (patch_size - 1) / 2.0
    z_c, y_c, x_c = torch.meshgrid(coords, coords, coords, indexing='ij')
 
    if shape_type == "random":
        shapes = ["cube", "sphere", "diamond"]
        shape_type = shapes[torch.randint(0, len(shapes), (1,)).item()]
 
    if shape_type == "sphere":
        radius = patch_size / 2.0
        dist_sq = (x_c - center)**2 + (y_c - center)**2 + (z_c - center)**2
        shape_template = dist_sq <= (radius**2)
    elif shape_type == "diamond":
        radius = patch_size / 2.0
        dist_manhattan = torch.abs(x_c - center) + torch.abs(y_c - center) + torch.abs(z_c - center)
        shape_template = dist_manhattan <= radius
    else: # cube
        shape_template = torch.ones((patch_size, patch_size, patch_size), dtype=torch.bool, device=x.device)
 
    # repeating the geometrical stamp over the grid
    global_shape_pattern = shape_template.repeat(grid_D, grid_H, grid_W)
    global_shape_pattern = global_shape_pattern.unsqueeze(0).expand(B, -1, -1, -1)
 
    # intersection: only applying the geometry in the selected patches
    voxel_mask = voxel_mask_base.bool() & global_shape_pattern
 
    # cropping to the original dimensions
    if symmetric:
        voxel_mask = voxel_mask[:, pad_left : pad_left + D, :H, :W]
    else:
        voxel_mask = voxel_mask[:, :D, :H, :W]
 
    voxel_mask   = voxel_mask.unsqueeze(1).expand_as(x).bool()
    masked_input = x.clone()
    masked_input[voxel_mask] = 0.0
    return masked_input, voxel_mask
 
 
class FFT3DLoss(nn.Module):
    def __init__(self):
        super().__init__()

class FFT2DSliceLoss(nn.Module):
    """
    FFT en 2D sobre slices axiales centrales en lugar de FFT 3D completa.
    ~50x más rápido que rfftn sobre el volumen completo.
    """
    def __init__(self, n_slices=8):
        super().__init__()
        self.n_slices = n_slices  # how many slices we sample

    def forward(self, pred, target, mask):
        B, C, D, H, W = pred.shape
        
        # choosing axial central slices 
        center = D // 2
        half   = self.n_slices // 2
        indices = range(center - half, center + half)

        pred_slices   = pred[:, :, indices, :, :]    # [B, C, n_slices, H, W]
        target_slices = target[:, :, indices, :, :]
        mask_slices   = mask[:, :, indices, :, :]

        # flattening slices in the patch to process them all at the same time
        pred_2d   = (pred_slices * mask_slices).view(B * C * self.n_slices, H, W)
        target_2d = (target_slices * mask_slices).view(B * C * self.n_slices, H, W)

        pred_fft   = torch.fft.rfft2(pred_2d,   norm="ortho")
        target_fft = torch.fft.rfft2(target_2d, norm="ortho")

        pred_abs   = torch.sqrt(pred_fft.real**2 + pred_fft.imag**2 + 1e-8)
        target_abs = torch.sqrt(target_fft.real**2 + target_fft.imag**2 + 1e-8)

        return F.l1_loss(pred_abs, target_abs)
    
# Main training
def run_experiment(config, lista_tensores, use_cluster=False):
    """Ejecuta un experimento de entrenamiento completo."""
 
    # constants of the experiment
    FORCE_TISSUE = config["force_tissue"]
    EXCLUDE_BG   = config["exclude_bg"]
    BATCH_SIZE   = config["batch_size"]
    LEARNING_RATE = config["lr"]
    LOSS_TYPE    = config["loss_type"]
    MASK_RATIO   = config["mask_ratio"]
    PATCH_SIZE   = config["patch_size"]
    SYMMETRIC    = config["symmetric"]
 
    MASK_RATIO, PATCH_SIZE = get_curriculum_params(0)
 
    EFFECTIVE_BATCH_SIZE = 4
    ACCUMULATION_STEPS   = max(1, EFFECTIVE_BATCH_SIZE // BATCH_SIZE)
    COMPATIBLE_SIZE      = (128, 160, 128)
    TOTAL_EPOCHS         = 600
 
    CHECKPOINT_DIR = (
        "./"
        f"checkpoints/1_CURRICULUM_L2_long_shapes_COSINE/{LOSS_TYPE}/checkpoints"
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
 
    print(f"\nJOB: {LOSS_TYPE}")
    print(f"BATCH size: {BATCH_SIZE} | Accumulation steps: {ACCUMULATION_STEPS}")
 
    # model, optimizer and scheduler
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = ResNet50_3D_MAE(in_chans=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)
    scaler    = torch.amp.GradScaler("cuda")
 
    # loss function
    if LOSS_TYPE == "mse":
        criterion = nn.MSELoss(reduction="none")
    elif LOSS_TYPE == "ssim":
        criterion = SSIM3DMapLoss(window_size=5, val_range=1.0).to(device)
    elif LOSS_TYPE == "ssim_l1":
        criterion    = SSIM3DMapLoss(window_size=5, val_range=1.0).to(device)
        criterion_l1 = nn.L1Loss(reduction="none").to(device)

    elif LOSS_TYPE == "fft": 
        criterion     = SSIM3DMapLoss(window_size=5, val_range=1.0).to(device)
        criterion_l1  = nn.L1Loss(reduction="none").to(device)
        criterion_fft = FFT2DSliceLoss(n_slices=8).to(device)

    elif LOSS_TYPE == "ms-ssim-true":
        criterion    = FusedMSSSIM3DMapLoss(padding="same").to(device)
        criterion_l1 = nn.L1Loss(reduction="none").to(device)

    else:
        raise ValueError(f"Loss '{LOSS_TYPE}' no soportado.")
 
    # loading last checkpoint
    START_EPOCH = 0
    if os.path.exists(LAST_CHECKPOINT_PATH):
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        START_EPOCH = checkpoint["epoch"] + 1
        print(f"Reanudando desde epoch {START_EPOCH}")
 
    try:
        wandb.init(
            project="Neurocosas_MAE_res_detail_loss",
            name=LOSS_TYPE,
            config=config,
        )
        wandb.watch(model, log="all", log_freq=50)
 
        # dataloader
        train_dataset = TensorDataset(tensor_paths=lista_tensores, compatible_size=COMPATIBLE_SIZE)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
 
        # training loop
        for epoch in range(START_EPOCH, TOTAL_EPOCHS):
            model.train()
            epoch_loss   = 0.0
            total_samples = 0
            epoch_z_sum, epoch_z_sq_sum, epoch_z_norm_sum = 0, 0, 0
 
            # dynamic curriculum
            MASK_RATIO, PATCH_SIZE = get_curriculum_params(epoch)
 
            progress_bar = tqdm(train_loader, desc=f"Ep {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
 
            for i, batch_images in enumerate(progress_bar):
                images = batch_images.to(device)
 
                masked_input, voxel_mask = mask_patches(
                    images, PATCH_SIZE, MASK_RATIO,
                    force_tissue_ratio=FORCE_TISSUE,
                    symmetric=SYMMETRIC,
                )
 
                with torch.amp.autocast("cuda"):
                    reconstruction, z = model(masked_input)
                    mask_float = voxel_mask.float()
 
                    # loss function, excluding background
                    if EXCLUDE_BG:
                        tissue_mask = (images > 0).float()
                        loss_mask   = mask_float * tissue_mask
                    else:
                        loss_mask = mask_float
 
                    # loss function calculation
                    if LOSS_TYPE == "mse":
                        loss_map = criterion(reconstruction, images)
                        loss     = (loss_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
 
                    elif LOSS_TYPE == "ssim":
                        ssim_map = criterion(reconstruction, images)
                        loss     = (ssim_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)

                    elif LOSS_TYPE == "fft": 
                        # 1. Calcular los mapas de error individuales
                        ssim_3d_map = criterion(reconstruction, images)
                        l1_map      = criterion_l1(reconstruction, images)
                        
                        # 2. Reducción con la máscara para SSIM y L1
                        loss_ssim   = (ssim_3d_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        loss_l1     = (l1_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        
                        # 3. Pérdida de frecuencia directa (Pasa la máscara por parámetro)
                        loss_fft = criterion_fft(reconstruction, images, loss_mask)
                        
                        # 4. Combinación híbrida final
                        loss = 0.70 * loss_ssim + 0.15 * loss_l1 + 0.15 * loss_fft
 
                    elif LOSS_TYPE == "ssim_l1":
                        ssim_3d_map = criterion(reconstruction, images)
                        l1_map      = criterion_l1(reconstruction, images)
                        loss_ssim   = (ssim_3d_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        loss_l1     = (l1_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        loss        = 0.84 * loss_ssim + 0.16 * loss_l1

                    elif LOSS_TYPE == "ms-ssim-true":
                        ssim_3d_map = criterion(reconstruction, images)
                        l1_map      = criterion_l1(reconstruction, images)
                        loss_ssim   = (ssim_3d_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        loss_l1     = (l1_map * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                        loss        = loss_ssim 
 
                    loss = loss / ACCUMULATION_STEPS
 
                scaler.scale(loss).backward()
 
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
 
                # Métricas de batch
                current_loss_val = loss.item() * ACCUMULATION_STEPS
                epoch_loss      += current_loss_val
 
                with torch.no_grad():
                    z_detached      = z.detach()
                    total_samples  += z_detached.shape[0]
                    epoch_z_sum    += z_detached.sum(dim=0)
                    epoch_z_sq_sum += (z_detached ** 2).sum(dim=0)
                    epoch_z_norm_sum += z_detached.norm(dim=1).sum().item()
 
                if i % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{current_loss_val:.4f}"})
 
            # end of epoch
            avg_loss  = epoch_loss / len(train_loader)
            scheduler.step()

            mean_z = epoch_z_sum / total_samples
            var_z  = torch.clamp((epoch_z_sq_sum / total_samples) - (mean_z ** 2), min=0.0)
            z_std  = torch.sqrt(var_z).mean().item()

            # just to log in wandb
            wandb.log({
                "Loss/train":            avg_loss,
                "Collapse/z_std":        z_std,
                "LearningRate":          scheduler.get_last_lr()[0],
                "Curriculum/mask_ratio": MASK_RATIO,  
                "Curriculum/patch_size": PATCH_SIZE,  
                "epoch":                 epoch,
            })
 
            if (epoch + 1) % 10 == 0:
                checkpoint_data = {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save(checkpoint_data, epoch_ckpt_path)
                torch.save(checkpoint_data, LAST_CHECKPOINT_PATH)

                # visualization
                model.eval()
                with torch.no_grad():
                    # using the last batch of the epoch
                    original_sample      = images.cpu()
                    masked_sample        = masked_input.cpu()
                    reconstructed_sample = reconstruction.detach().cpu()

                    # adding .float() before .numpy() to avoid a float16 error
                    nib.save(
                        nib.Nifti1Image(original_sample[0, 0].float().numpy(), np.eye(4)),
                        os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}_01_original.nii.gz")
                    )
                    nib.save(
                        nib.Nifti1Image(masked_sample[0, 0].float().numpy(), np.eye(4)),
                        os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}_02_masked.nii.gz")
                    )
                    nib.save(
                        nib.Nifti1Image(reconstructed_sample[0, 0].float().numpy(), np.eye(4)),
                        os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}_03_reconstructed.nii.gz")
                    )

                    # three views of the first volume of the batch
                    D, H, W = COMPATIBLE_SIZE
                    sl_ax  = original_sample[0, 0, D // 2, :, :]   
                    sl_cor = original_sample[0, 0, :, H // 2, :]   
                    sl_sag = original_sample[0, 0, :, :, W // 2]   

                    def to_wandb_image(tensor_2d, caption):
                        arr = tensor_2d.numpy()
                        # normalizing to [0, 255] to achieve the correct visualization
                        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                        return wandb.Image((arr * 255).astype(np.uint8), caption=caption)

                    wandb.log({
                        "Visualizacion/Axial": [
                            to_wandb_image(original_sample[0, 0, D // 2, :, :],        "Original"),
                            to_wandb_image(masked_sample[0, 0, D // 2, :, :],          "Masked"),
                            to_wandb_image(reconstructed_sample[0, 0, D // 2, :, :],   "Reconstructed"),
                        ],
                        "Visualizacion/Coronal": [
                            to_wandb_image(original_sample[0, 0, :, H // 2, :],        "Original"),
                            to_wandb_image(masked_sample[0, 0, :, H // 2, :],          "Masked"),
                            to_wandb_image(reconstructed_sample[0, 0, :, H // 2, :],   "Reconstructed"),
                        ],
                        "Visualizacion/Sagital": [
                            to_wandb_image(original_sample[0, 0, :, :, W // 2],        "Original"),
                            to_wandb_image(masked_sample[0, 0, :, :, W // 2],          "Masked"),
                            to_wandb_image(reconstructed_sample[0, 0, :, :, W // 2],   "Reconstructed"),
                        ],
                        "epoch": epoch,
                    })
                model.train()
 
        wandb.finish()
 
    except Exception:
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish(exit_code=1)
 
 
# Entry point
if __name__ == "__main__":
 
    CSV_PREPROCESADO = (
        "./PREPROCESSED_TENSORS_128x160x128_ALL/preprocessed_dataset.csv"
    )
    df_tensores   = pd.read_csv(CSV_PREPROCESADO)
    lista_tensores = df_tensores["tensor_path"].tolist()
    print(f"Loading {len(lista_tensores)} preprocessed tensors")
 
    USE_CLUSTER = 0  # 0 = local, 1 = cluster
 
    # local configuration
    config = {
        "base_name":    "FINAL_local_test_large_shapes_CURRICULUM",
        "batch_size":   2,
        "lr":           1e-4,
        "loss_type":    "ms-ssim-true",
        "mask_ratio":   0.75, #ignored
        "patch_size":   16, #ignored
        "force_tissue": True,
        "exclude_bg":   True,
        "symmetric":    True,
    }
 
    if USE_CLUSTER == 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=int, required=True, help="Índice del experimento")
        args = parser.parse_args()
 
        lista_losses = ["ssim", "ms-ssim", "mse", "fft"]
        config["loss_type"] = lista_losses[args.config]
        print(f"Loaded configuration for cluster. Evaluating loss: {config['loss_type']}")
        print(f"Configuration {args.config} cloaded in the cluster")
    else:
        print("running on local gpu")
 
    run_experiment(config, lista_tensores, use_cluster=bool(USE_CLUSTER))
