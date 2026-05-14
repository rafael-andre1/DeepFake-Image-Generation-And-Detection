#!/usr/bin/env python
# coding: utf-8

# # GAN — g5 full-data EMA + best-FID checkpoint
# 
# Nesta fase deixamos de testar apenas alterações isoladas e passamos a construir a **melhor versão final do gerador** com base no que já aprendemos nas fases anteriores.
# 
# Resumo das conclusões até agora:
# 
# - **g0 baseline:** DCGAN mensurável, mas FID alto e imagens muito ruidosas.
# - **g1 more data:** aumentar folds reais foi a maior melhoria observada.
# - **g2 discriminator balancing:** reduzir `LR_D` + label smoothing deixou o discriminador menos saturado, mas piorou FID/KID.
# - **g3 EMA:** EMA no gerador melhorou ligeiramente o FID e estabilizou samples.
# - **g4 SpectralNorm D:** regularizou o discriminador, mas piorou FID e imagens.
# 
# Decisão para o g5:
# 
# > Usar a melhor combinação observada até agora: **mais dados + EMA**, sem SpectralNorm, sem label smoothing e sem reduzir o learning rate do discriminador.
# 
# Alteração principal desta fase:
# 
# - partir do g3;
# - aumentar os folds reais usados no treino, idealmente para `START_FOLD = 0`, `END_FOLD = 99`;
# - manter `USE_EMA = True`;
# - guardar automaticamente o checkpoint com **melhor FID**, porque o melhor gerador pode não ser o da última epoch.
# 
# Objetivo:
# 
# > Obter o melhor gerador funcional possível para criar imagens de pessoas, usando o checkpoint com menor FID como modelo final.
# 

# In[ ]:


import os
import json
import base64
import io
import math
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid


import sys
sys.path.append(str(Path.cwd().parent))

from utils import DeepFakeDataset, dcganFormat  # <-- your utils module


# ## Config

# In[ ]:


# =========================================================
# Config — g5 full-data EMA + best-FID checkpoint
# =========================================================
# Esta experiência parte do g3_ema_generator e tenta obter o melhor gerador final.
# Mantemos o que funcionou: mais dados + EMA.
# Não usamos SpectralNorm, label smoothing nem redução do LR do discriminador.

RUN_NAME = "g5_full_data_ema_bestfid"

REAL_DIR = "../../deepfake_data/wiki"
BASE_OUTPUT_DIR = "outputs"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

IMAGE_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 100
NGF = 64
NDF = 64
NUM_CHANNELS = 3

# Parâmetros de treino — mesma base do g1/g3
NUM_EPOCHS = 200
LR_G = 2e-4
LR_D = 1e-4
BETA1 = 0.5
SEED = 42
NUM_WORKERS = 0

# Sem label smoothing: g2 mostrou que enfraquecer D desta forma piorou FID.
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# Principal alteração do g5: usar mais dados reais.
# Se END_FOLD = 99 for demasiado pesado, usar 50 numa primeira execução.
START_FOLD = 0
END_FOLD = 99
INTERVAL = True

# Logging / avaliação
SAMPLE_EVERY = 25          # guardar PNG com fixed_noise a cada N epochs
CHECKPOINT_EVERY = 10      # checkpoints regulares de segurança
FID_EVERY = 50             # calcular FID/KID a cada N epochs
FID_NUM_IMAGES = 300       # usar 300 se houver problemas de memória; 1000+ se houver tempo/VRAM
SAVE_JSONL_EVERY = None    # usar 1 para JSONL por epoch; None desliga JSONL
COMPUTE_KID = True         # se ficar lento ou pesado, mudar para False

# EMA do gerador: usado para samples, FID/KID, best checkpoint e interpolação.
USE_EMA = True
EMA_DECAY = 0.999

# Guardar sempre estes epochs, além dos intervalos.
ALWAYS_SAVE_EPOCHS = {1, NUM_EPOCHS}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

# Criar pastas dos outputs desta run
for folder in [OUTPUT_DIR, SAMPLES_DIR, CHECKPOINTS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Reprodutibilidade
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Mantém alguma performance. Para reprodutibilidade máxima, pôr benchmark=False e deterministic=True.
torch.backends.cudnn.benchmark = True

print("Run:", RUN_NAME)
print("Device:", DEVICE)
print("Output dir:", OUTPUT_DIR)
print("Folds:", START_FOLD, "->", END_FOLD)


# ## Utils

# In[ ]:


# =========================================================
# Utils de logging, samples e métricas
# =========================================================
from PIL import Image as PILImage


def denorm_to_01(x):
    """Converte tensores de [-1, 1] para [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


def to_uint8_images(x):
    """Converte batch de imagens [-1,1] ou [0,1] para uint8 [0,255]."""
    if x.min() < 0:
        x = denorm_to_01(x)
    return (x * 255).clamp(0, 255).to(torch.uint8)


def should_run(epoch, every):
    if every is None:
        return False
    return epoch in ALWAYS_SAVE_EPOCHS or epoch % every == 0


def save_generated_samples_png(epoch, generator, noise, out_dir=SAMPLES_DIR, prefix="fixed_noise", nrow=8):
    """Guarda grelha PNG usando o fixed_noise."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        fake_images = denorm_to_01(generator(noise).detach().cpu())
        grid = make_grid(fake_images, nrow=nrow, padding=2, normalize=False)
        path = os.path.join(out_dir, f"{prefix}_epoch_{epoch:03d}.png")
        save_image(grid, path)
    if was_training:
        generator.train()
    return path


def save_samples_jsonl(epoch, generator, noise, path, nrow=8):
    """Append one lossless-PNG sample grid (base64) as a JSON line."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        fake = denorm_to_01(generator(noise).detach().cpu())
        grid = make_grid(fake, nrow=nrow, padding=2, normalize=False)
        arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if was_training:
        generator.train()

    with open(path, "a") as f:
        f.write(json.dumps({"epoch": epoch, "format": "png", "grid_b64": b64}) + "\n")


def save_latent_interpolation(generator, latent_dim, device, out_path, steps=8, nrow=8):
    """Guarda uma interpolação linear entre dois vetores latentes."""
    was_training = generator.training
    generator.eval()
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    alphas = torch.linspace(0, 1, steps=steps, device=device)
    z_interp = torch.cat([(1 - a) * z1 + a * z2 for a in alphas], dim=0)
    with torch.no_grad():
        imgs = denorm_to_01(generator(z_interp).detach().cpu())
        grid = make_grid(imgs, nrow=nrow, padding=2, normalize=False)
        save_image(grid, out_path)
    if was_training:
        generator.train()
    return out_path


def save_real_grid(dataloader, out_path, n=16, nrow=4):
    """Guarda uma grelha de imagens reais para referência visual."""
    batch = next(iter(dataloader))[:n].cpu()
    batch = denorm_to_01(batch)
    grid = make_grid(batch, nrow=nrow, padding=2, normalize=False)
    save_image(grid, out_path)
    return out_path


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Atualiza os pesos EMA do gerador: ema = decay*ema + (1-decay)*model."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    # Mantém buffers como running_mean/running_var de BatchNorm coerentes com o gerador atual.
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


# ---- FID / KID opcionais ----
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TORCHMETRICS_AVAILABLE = True
    print("torchmetrics disponível: FID/KID serão calculados.")
except Exception as e:
    FrechetInceptionDistance = None
    KernelInceptionDistance = None
    TORCHMETRICS_AVAILABLE = False
    print("torchmetrics não disponível: FID/KID serão ignorados.")
    print("Para ativar: pip install torchmetrics[image] torch-fidelity")
    print("Erro:", repr(e))


@torch.no_grad()
def compute_fid_kid(
    generator,
    dataloader,
    num_images=1024,
    latent_dim=LATENT_DIM,
    device=DEVICE,
    compute_kid=False,
):
    """Calcula FID e, opcionalmente, KID com torchmetrics.

    Para evitar bloqueios/memória excessiva no VS Code/Jupyter, o KID fica
    desligado por defeito. Quando compute_kid=False, a função nem instancia
    KernelInceptionDistance, evitando o buffer grande de features.
    """
    import gc

    if not TORCHMETRICS_AVAILABLE:
        return {"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")}

    was_training = generator.training
    generator.eval()

    n_target = min(int(num_images), len(dataloader.dataset))
    if n_target <= 0:
        if was_training:
            generator.train()
        return {"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")}

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = None
    if compute_kid:
        # subset_size tem de ser <= número de imagens usadas.
        kid = KernelInceptionDistance(subset_size=min(50, n_target)).to(device)

    # Imagens reais
    seen = 0
    for real in dataloader:
        if isinstance(real, (list, tuple)):
            real = real[0]
        real = real.to(device, non_blocking=True)

        take = min(real.size(0), n_target - seen)
        if take <= 0:
            break

        real_u8 = to_uint8_images(real[:take])
        fid.update(real_u8, real=True)
        if kid is not None:
            kid.update(real_u8, real=True)
        seen += take

        del real, real_u8

    # Imagens falsas
    seen = 0
    while seen < n_target:
        b = min(BATCH_SIZE, n_target - seen)
        noise = torch.randn(b, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        fake_u8 = to_uint8_images(fake)

        fid.update(fake_u8, real=False)
        if kid is not None:
            kid.update(fake_u8, real=False)
        seen += b

        del noise, fake, fake_u8

    fid_value = float(fid.compute().detach().cpu())

    if kid is not None:
        kid_mean, kid_std = kid.compute()
        kid_mean = float(kid_mean.detach().cpu())
        kid_std = float(kid_std.detach().cpu())
    else:
        kid_mean = float("nan")
        kid_std = float("nan")

    del fid, kid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if was_training:
        generator.train()

    return {
        "fid": fid_value,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }


# ## Dataset & DataLoader

# In[ ]:


dataset = DeepFakeDataset(
    img_dir=REAL_DIR,
    label=1,                           # ignored; image_only=True
    transform=dcganFormat(IMAGE_SIZE),
    range_folds=[START_FOLD, END_FOLD],
    interval=INTERVAL,
    image_only=True,                   # GAN mode: return only image tensor
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,
)

print(f"Total de imagens reais: {len(dataset)}")
print(f"Folds usados: {dataset.fold_names}")

# Guardar grelha de imagens reais para comparação visual
real_grid_path = save_real_grid(dataloader, os.path.join(SAMPLES_DIR, "real_reference_grid.png"))
print("Grelha real guardada em:", real_grid_path)


# ### Ver uma imagem do dataset

# In[ ]:


dataset.show(0)


# ### Inicialização de pesos

# In[ ]:


# Esta função inicializa os pesos da rede, se for:
# - Camada convolucional: inicializa os pesos com distribuição normal (média 0 e desvio padrão 0.02)
# - Camada BatchNorm: pesos perto de 1 e bias 0 
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Isto é o tipo de inicialização clássica em DCGAN


# # Gerador

# In[ ]:


# O Gerador recebe um vetor aleatório z e tenta convertê-lo numa imagem falsa realista
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()

        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.init(z)
        return self.main(x)


# # Discriminador

# In[ ]:


# Recebe uma imagem e tenta dizer se ela é real ou falsa
# Ele faz o oposto do gerador, reduz progressivamente a resolução, aumentando profundidade e no fim produz um único valor por imagem
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x).view(-1)


# # Instanciar modelos

# In[ ]:


netG = Generator(LATENT_DIM, NGF, NUM_CHANNELS).to(DEVICE)
netD = Discriminator(NUM_CHANNELS, NDF).to(DEVICE)

netG.apply(weights_init)
netD.apply(weights_init)

# EMA começa como uma cópia exata do gerador inicial.
# Só é usado para avaliação/amostragem; não recebe gradientes.
netG_ema = copy.deepcopy(netG).to(DEVICE).eval()
for p in netG_ema.parameters():
    p.requires_grad_(False)

print(netG)
print(netD)
print("EMA generator active:", USE_EMA)


# In[ ]:


# Loss binária com logits
# O discriminador devolve logits crus, sem sigmoid, esta loss combina sigmoid e binary cross entropy de forma mais estavel numericamente
criterion = nn.BCEWithLogitsLoss()

# Otimizadores Adam para o discriminador e o gerador
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))

# Ruido fixo para as amostras. Este ruido serve para gerar sempre as mesmas 64 imagens ao longo do treino.
# Deste modo podemos ver a evolução do gerador de forma consistente (na epoch 1, 10, 50, etc)
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)


# ## Função para guardar amostras

# In[ ]:


# Wrapper compatível com versões anteriores do notebook
def save_generated_samples(epoch, generator, noise):
    return save_generated_samples_png(epoch, generator, noise)


# ## Treino — guardar melhor checkpoint por FID

# In[ ]:


print("CUDA available:", torch.cuda.is_available())
print("netG on:", next(netG.parameters()).device)
print("netD on:", next(netD.parameters()).device)


# In[ ]:


from tqdm.auto import tqdm

# =========================================================
# Treino — g5 full-data EMA + best-FID checkpoint
# =========================================================
# Parte do g3_ema_generator. O treino adversarial continua igual;
# aumentamos os dados reais e guardamos automaticamente o melhor checkpoint por FID.

history = []
fid_history = []
g_losses, d_losses = [], []

history_path = os.path.join(METRICS_DIR, "training_history.csv")
fid_path = os.path.join(METRICS_DIR, "fid_kid_history.csv")
config_path = os.path.join(OUTPUT_DIR, "run_config.json")

run_config = {
    "run_name": RUN_NAME,
    "real_dir": REAL_DIR,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "latent_dim": LATENT_DIM,
    "ngf": NGF,
    "ndf": NDF,
    "num_epochs": NUM_EPOCHS,
    "lr_g": LR_G,
    "lr_d": LR_D,
    "beta1": BETA1,
    "real_label": REAL_LABEL,
    "fake_label": FAKE_LABEL,
    "use_ema": USE_EMA,
    "ema_decay": EMA_DECAY,
    "best_checkpoint_metric": "fid",
    "final_model_selection": "best_fid_checkpoint",
    "seed": SEED,
    "start_fold": START_FOLD,
    "end_fold": END_FOLD,
    "interval": INTERVAL,
    "sample_every": SAMPLE_EVERY,
    "checkpoint_every": CHECKPOINT_EVERY,
    "fid_every": FID_EVERY,
    "fid_num_images": FID_NUM_IMAGES,
    "compute_kid": COMPUTE_KID,
    "num_workers": NUM_WORKERS,
    "num_real_images": len(dataset),
    "folds_used": list(dataset.fold_names),
}
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)

# O modelo final será escolhido pelo melhor FID observado, não obrigatoriamente pela última epoch.
best_fid = float("inf")
best_fid_epoch = None
best_checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_fid_checkpoint.pt")
best_sample_prefix = "best_fid_fixed_noise"

# JSONL opcional para evolução detalhada por epoch
if SAVE_JSONL_EVERY is not None:
    samples_jsonl_path = os.path.join(OUTPUT_DIR, "samples_evolution.jsonl")
    open(samples_jsonl_path, "w").close()
else:
    samples_jsonl_path = None

def eval_generator():
    """Gerador usado para avaliação: EMA no g3, G normal se USE_EMA=False."""
    return netG_ema if USE_EMA else netG

# Guardar samples iniciais antes do treino
save_generated_samples_png(0, eval_generator(), fixed_noise, prefix="fixed_noise_initial")

print("A começar treino g5...")
epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training", unit="epoch")

for epoch in epoch_bar:
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_dx = 0.0
    epoch_dgz_fake = 0.0
    epoch_dgz_gen = 0.0

    batch_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
    for real_images in batch_bar:
        real_images = real_images.to(DEVICE)
        b_size = real_images.size(0)

        # --- Discriminator ---
        netD.zero_grad(set_to_none=True)
        output_real = netD(real_images)
        real_targets = torch.full((b_size,), REAL_LABEL, device=DEVICE)
        lossD_real = criterion(output_real, real_targets)

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        fake_targets = torch.full((b_size,), FAKE_LABEL, device=DEVICE)
        lossD_fake = criterion(output_fake, fake_targets)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # --- Generator ---
        netG.zero_grad(set_to_none=True)
        output_gen = netD(fake_images)
        gen_targets = torch.full((b_size,), REAL_LABEL, device=DEVICE)
        lossG = criterion(output_gen, gen_targets)
        lossG.backward()
        optimizerG.step()

        # Atualizar EMA do gerador depois de cada update de G.
        if USE_EMA:
            update_ema(netG_ema, netG, EMA_DECAY)

        # Métricas por batch
        with torch.no_grad():
            dx = torch.sigmoid(output_real).mean().item()
            dgz_fake = torch.sigmoid(output_fake).mean().item()
            dgz_gen = torch.sigmoid(output_gen).mean().item()

        epoch_d_loss += lossD.item()
        epoch_g_loss += lossG.item()
        epoch_dx += dx
        epoch_dgz_fake += dgz_fake
        epoch_dgz_gen += dgz_gen

        batch_bar.set_postfix(d=f"{lossD.item():.3f}", g=f"{lossG.item():.3f}", dx=f"{dx:.2f}")

    n = len(dataloader)
    avg_d = epoch_d_loss / n
    avg_g = epoch_g_loss / n
    avg_dx = epoch_dx / n
    avg_dgz_fake = epoch_dgz_fake / n
    avg_dgz_gen = epoch_dgz_gen / n

    d_losses.append(avg_d)
    g_losses.append(avg_g)

    row = {
        "run_name": RUN_NAME,
        "epoch": epoch,
        "loss_d": avg_d,
        "loss_g": avg_g,
        "D_x": avg_dx,
        "D_G_z_fake": avg_dgz_fake,
        "D_G_z_gen": avg_dgz_gen,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(history_path, index=False)

    epoch_bar.set_postfix(loss_d=f"{avg_d:.3f}", loss_g=f"{avg_g:.3f}", d_x=f"{avg_dx:.2f}", dgz=f"{avg_dgz_gen:.2f}")

    # Samples PNG selecionadas — no g5 usamos EMA.
    if should_run(epoch, SAMPLE_EVERY):
        save_generated_samples_png(epoch, eval_generator(), fixed_noise, prefix="fixed_noise")

    # JSONL opcional — no g5 usamos EMA.
    if SAVE_JSONL_EVERY is not None and should_run(epoch, SAVE_JSONL_EVERY):
        save_samples_jsonl(epoch, eval_generator(), fixed_noise, samples_jsonl_path)

    # FID/KID selecionado — no g5 avaliamos EMA e guardamos o melhor FID.
    if FID_EVERY is not None and (epoch % FID_EVERY == 0 or epoch == NUM_EPOCHS):
        metrics = compute_fid_kid(eval_generator(), dataloader, num_images=FID_NUM_IMAGES, compute_kid=COMPUTE_KID)
        metrics.update({"run_name": RUN_NAME, "epoch": epoch, "evaluated_model": "ema" if USE_EMA else "raw"})
        fid_history.append(metrics)
        pd.DataFrame(fid_history).to_csv(fid_path, index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_fid = metrics.get("fid")
        if current_fid is not None and pd.notna(current_fid) and current_fid < best_fid:
            best_fid = float(current_fid)
            best_fid_epoch = epoch

            best_checkpoint = {
                "epoch": epoch,
                "best_fid": best_fid,
                "best_fid_epoch": best_fid_epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "g_losses": g_losses,
                "d_losses": d_losses,
                "history": history,
                "fid_history": fid_history,
                "config": run_config,
            }
            if USE_EMA:
                best_checkpoint["netG_ema_state_dict"] = netG_ema.state_dict()

            torch.save(best_checkpoint, best_checkpoint_path)
            save_generated_samples_png(epoch, eval_generator(), fixed_noise, prefix=best_sample_prefix)
            print(f"Novo melhor FID: {best_fid:.3f} @ epoch {epoch:03d}. Checkpoint guardado em: {best_checkpoint_path}")

        if COMPUTE_KID:
            print(f"Epoch {epoch:03d} | FID: {metrics['fid']:.3f} | KID: {metrics['kid_mean']:.5f} ± {metrics['kid_std']:.5f} | model: {'EMA' if USE_EMA else 'raw'}")
        else:
            print(f"Epoch {epoch:03d} | FID: {metrics['fid']:.3f} | KID: desligado | model: {'EMA' if USE_EMA else 'raw'}")

    # Checkpoints selecionados
    if should_run(epoch, CHECKPOINT_EVERY):
        checkpoint = {
            "epoch": epoch,
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimizerG_state_dict": optimizerG.state_dict(),
            "optimizerD_state_dict": optimizerD.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
            "history": history,
            "fid_history": fid_history,
            "config": run_config,
        }
        if USE_EMA:
            checkpoint["netG_ema_state_dict"] = netG_ema.state_dict()
        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, f"dcgan_{RUN_NAME}_epoch_{epoch:03d}.pt"))

# Artefactos finais — no g5 usamos EMA.
interpolation_path = save_latent_interpolation(
    eval_generator(),
    LATENT_DIM,
    DEVICE,
    os.path.join(SAMPLES_DIR, "latent_interpolation_final.png"),
    steps=8,
    nrow=8,
)
print("Treino terminado.")
print("Histórico:", history_path)
print("FID/KID:", fid_path)
print("Interpolação latente:", interpolation_path)
print("Melhor FID:", best_fid, "@ epoch", best_fid_epoch)
print("Best checkpoint:", best_checkpoint_path)


# In[ ]:


import matplotlib.pyplot as plt

history_path = os.path.join(METRICS_DIR, "training_history.csv")
history_df = pd.read_csv(history_path)

plt.figure(figsize=(10, 5))
plt.plot(history_df["epoch"], history_df["loss_g"], label="Generator Loss")
plt.plot(history_df["epoch"], history_df["loss_d"], label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"GAN Training Losses — {RUN_NAME}")
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(PLOTS_DIR, "loss_curves.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_df["epoch"], history_df["D_x"], label="D(x)")
plt.plot(history_df["epoch"], history_df["D_G_z_fake"], label="D(G(z)) fake")
plt.plot(history_df["epoch"], history_df["D_G_z_gen"], label="D(G(z)) gen")
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.title(f"Discriminator Outputs During Training — {RUN_NAME}")
plt.legend()
plt.grid(True)
d_plot_path = os.path.join(PLOTS_DIR, "discriminator_outputs.png")
plt.savefig(d_plot_path, dpi=150, bbox_inches="tight")
plt.show()

fid_path = os.path.join(METRICS_DIR, "fid_kid_history.csv")
if os.path.exists(fid_path):
    fid_df = pd.read_csv(fid_path)
    if "fid" in fid_df and fid_df["fid"].notna().any():
        plt.figure(figsize=(10, 5))
        plt.plot(fid_df["epoch"], fid_df["fid"], marker="o", label="FID")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.title(f"FID vs Epoch — {RUN_NAME}")
        plt.legend()
        plt.grid(True)
        fid_plot_path = os.path.join(PLOTS_DIR, "fid_curve.png")
        plt.savefig(fid_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
    else:
        print("FID existe, mas está vazio/NaN. Verificar torchmetrics/torch-fidelity.")
else:
    print("Ainda não existe ficheiro FID/KID. Corre o treino primeiro.")

print("Plots guardados em:", PLOTS_DIR)


# In[ ]:


fid_path = os.path.join(METRICS_DIR, "fid_kid_history.csv")

if os.path.exists(fid_path):
    fid_df = pd.read_csv(fid_path)

    # --- FID plot ---
    if "fid" in fid_df.columns and fid_df["fid"].notna().any():
        plt.figure(figsize=(10, 5))
        plt.plot(fid_df["epoch"], fid_df["fid"], marker="o", label="FID")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.title(f"FID vs Epoch — {RUN_NAME}")
        plt.legend()
        plt.grid(True)

        fid_plot_path = os.path.join(PLOTS_DIR, "fid_curve.png")
        plt.savefig(fid_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
    else:
        print("FID existe, mas está vazio/NaN. Verificar torchmetrics/torch-fidelity.")

    # --- KID plot, only if available ---
    if "kid_mean" in fid_df.columns and fid_df["kid_mean"].notna().any():
        plt.figure(figsize=(10, 5))
        plt.plot(fid_df["epoch"], fid_df["kid_mean"], marker="o", label="KID mean")

        if "kid_std" in fid_df.columns and fid_df["kid_std"].notna().any():
            lower = fid_df["kid_mean"] - fid_df["kid_std"]
            upper = fid_df["kid_mean"] + fid_df["kid_std"]
            plt.fill_between(fid_df["epoch"], lower, upper, alpha=0.2, label="KID ± std")

        plt.xlabel("Epoch")
        plt.ylabel("KID")
        plt.title(f"KID vs Epoch — {RUN_NAME}")
        plt.legend()
        plt.grid(True)

        kid_plot_path = os.path.join(PLOTS_DIR, "kid_curve.png")
        plt.savefig(kid_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
    else:
        print("KID desligado ou sem valores válidos; a avaliação desta run usa FID como métrica principal.")

else:
    print("Ainda não existe ficheiro FID/KID. Corre o treino primeiro.")

print("Plots guardados em:", PLOTS_DIR)


# Como o D devolve logits, aplicas sigmoid para obter algo parecido com probabilidade.
# 
# - `D(x):` Probabilidade média atribuída às imagens reais. (idealmente alta)
# 
# - `D(G(z)) fake:` Probabilidade média atribuída às falsas quando treinas o D. (idealmente baixa)
# 
# - `D(G(z)) gen:` Probabilidade média atribuída às falsas quando observadas no passo do G. (se começar a subir, pode indicar que o G está a enganar melhor o D)

# # Gerar imagens a partir do último modelo em memória

# In[ ]:


# =========================================================
# Gerar samples com o MELHOR modelo (best FID checkpoint)
# =========================================================
import copy
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

best_ckpt_path = os.path.join(CHECKPOINTS_DIR, "best_fid_checkpoint.pt")
ckpt = torch.load(best_ckpt_path, map_location=DEVICE)

print("Best checkpoint epoch:", ckpt.get("epoch"))
print("Best FID:", ckpt.get("best_fid"))

# Criar uma cópia do modelo para não mexer no netG/netG_ema atual em memória
best_gen = copy.deepcopy(netG_ema if USE_EMA else netG)

# Se usas EMA e o checkpoint tiver netG_ema_state_dict, usar essa versão
if USE_EMA and ckpt.get("netG_ema_state_dict") is not None:
    best_gen.load_state_dict(ckpt["netG_ema_state_dict"])
else:
    best_gen.load_state_dict(ckpt["netG_state_dict"])

best_gen = best_gen.to(DEVICE)
best_gen.eval()

with torch.no_grad():
    noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)
    fake_images = denorm_to_01(best_gen(noise).cpu())

grid = make_grid(fake_images, nrow=4, padding=2)

plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()

manual_sample_path = os.path.join(SAMPLES_DIR, "manual_random_samples_best_fid.png")
save_image(grid, manual_sample_path)
print("Samples guardadas em:", manual_sample_path)

manual_gen = netG_ema if USE_EMA else netG
manual_gen.eval()
with torch.no_grad():
    noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)
    fake_images = denorm_to_01(manual_gen(noise).cpu())

grid = make_grid(fake_images, nrow=4, padding=2)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()

manual_sample_path = os.path.join(SAMPLES_DIR, "manual_random_samples.png")
save_image(grid, manual_sample_path)
print("Samples guardadas em:", manual_sample_path)

# # Confirmar que os dados estão bem:

# In[ ]:


import matplotlib.pyplot as plt
from torchvision.utils import make_grid

real_batch = next(iter(dataloader))
real_batch = denorm_to_01(real_batch[:16].cpu())

grid = make_grid(real_batch, nrow=4, padding=2)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()


# In[ ]:


real_batch = next(iter(dataloader))
img = denorm_to_01(real_batch[0].cpu())

import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.show()


# In[ ]:


# Resumo rápido da run para copiar para o JOURNAL.md
summary = {
    "Iteration": "g5_full_data_ema_bestfid",
    "Change": "Start from g3 EMA and increase real folds to 0–99; select final generator by best FID checkpoint instead of last epoch",
    "FID @ end": None,
    "Best FID": None,
    "Best FID epoch": None,
    "Notes": "Testa se usar mais dados reais, mantendo EMA e a configuração que funcionou melhor, melhora o gerador final de pessoas.",
}

fid_path = os.path.join(METRICS_DIR, "fid_kid_history.csv")
if os.path.exists(fid_path):
    fid_df = pd.read_csv(fid_path)
    if len(fid_df) > 0 and fid_df["fid"].notna().any():
        valid = fid_df.dropna(subset=["fid"])
        summary["FID @ end"] = float(valid.iloc[-1]["fid"])
        best_row = valid.sort_values("fid").iloc[0]
        summary["Best FID"] = float(best_row["fid"])
        summary["Best FID epoch"] = int(best_row["epoch"])

print(json.dumps(summary, indent=2, ensure_ascii=False))


# # Conclusão

# ## Conclusão — g5_full_data_ema_bestfid
# 
# Na fase g5, partimos da melhor configuração anterior, **g3_ema_generator**, e aumentámos o conjunto de treino para usar todos os folds reais disponíveis (`START_FOLD = 0`, `END_FOLD = 99`). Mantivemos a arquitetura DCGAN, os mesmos learning rates (`LR_G = 2e-4`, `LR_D = 1e-4`) e a EMA do gerador, sem reintroduzir label smoothing nem SpectralNorm, uma vez que essas alterações tinham piorado os resultados nas fases anteriores.
# 
# A principal alteração desta fase foi combinar **mais dados reais** com **seleção automática do melhor checkpoint por FID**. Em vez de assumir que a última epoch seria a melhor, o treino passou a guardar `best_fid_checkpoint.pt` sempre que era obtido um novo melhor FID.
# 
# Os resultados melhoraram significativamente face às versões anteriores. O melhor FID obtido foi aproximadamente **163.31 na epoch 100**, enquanto o FID final na epoch 200 ficou em aproximadamente **170.49**. Isto confirma que o melhor modelo não foi o último, mas sim o checkpoint intermédio selecionado por FID.
# 
# Comparando com as fases anteriores:
# 
# | Fase | Alteração principal | Melhor FID aproximado |
# |---|---:|---:|
# | g0 | DCGAN baseline | 300.4 |
# | g1 | mais dados, folds 0–20 | 209.0 |
# | g3 | g1 + EMA | 205.8 |
# | g5 | folds 0–99 + EMA + best checkpoint | **163.3** |
# 
# Visualmente, o gerador passou a produzir imagens com estrutura humana mais evidente, incluindo silhuetas, zonas de rosto, cabelo, roupa e fundos mais consistentes. No entanto, as imagens continuam longe de ser pessoas realistas: ainda há deformações, baixa definição facial, artefactos cromáticos e instabilidade na anatomia.
# 
# A curva de treino mostra que o gerador continua a ter dificuldade contra o discriminador, com a loss do gerador a crescer ao longo do treino. Ainda assim, a utilização de mais dados reais melhorou claramente a distribuição aprendida, e a seleção por melhor FID evitou escolher uma epoch final menos adequada.
# 
# **Decisão:** manter o g5 como melhor modelo até agora. O checkpoint a usar como modelo final provisório deve ser `best_fid_checkpoint.pt`, correspondente à melhor epoch por FID, e não o modelo da última epoch.

# ---
# 

# | g5 | full_data_ema_bestfid | Use g3 setup with all real folds (0–99) and save best checkpoint by FID | Best FID ≈ 163.31 @ epoch 100; final FID ≈ 170.49 | Best model so far. More real data gives the largest improvement, and best-FID checkpoint selection avoids using a worse final epoch. Samples are more human-like but still blurry/deformed. |
