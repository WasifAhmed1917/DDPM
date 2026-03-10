# -*- coding: utf-8 -*-
"""DVLM_PA1.ipynb

"""

import torch
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import diffusers
import tqdm
from tqdm.auto import tqdm
import scipy

"""## Task 0: Dataset Setup"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

"""## Tasks 1 and 2"""

class DDPM:
    def __init__(self, L=1000, beta_min=1e-4, beta_max=0.02, device='cuda'):
        self.L = L
        self.device = device

        self.betas = torch.linspace(beta_min, beta_max, L, device=device)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """(Eq 2)."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t     = self.sqrt_alpha_bars[t - 1].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t - 1].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def sample_timesteps(self, batch_size):
        """Samples i ~ Unif{1, ..., L}"""
        return torch.randint(1, self.L + 1, (batch_size,), device=self.device)

    def q_posterior_mean_var(self, x0, xi, i):
        beta_i         = self.betas[i - 1].view(-1, 1, 1, 1)
        alpha_i        = self.alphas[i - 1].view(-1, 1, 1, 1)
        alpha_bar_i    = self.alpha_bars[i - 1].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars_prev[i - 1].view(-1, 1, 1, 1)

        posterior_var  = beta_i * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_i)
        posterior_mean = (
            (alpha_bar_prev.sqrt() * beta_i / (1.0 - alpha_bar_i)) * x0
            + (alpha_i.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_i)) * xi
        )
        return posterior_mean, posterior_var

import matplotlib.pyplot as plt

def verify_schedule(diffusion):
    alphas_bar = diffusion.alpha_bars.cpu().numpy()
    steps = np.arange(1, diffusion.L + 1)

    snr = alphas_bar / (1 - alphas_bar)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps, alphas_bar)
    plt.title(r"$\bar{\alpha}_i$ Schedule")
    plt.xlabel("Timestep i")
    plt.ylabel(r"$\bar{\alpha}_i$")

    plt.subplot(1, 2, 2)
    plt.plot(steps, snr)
    plt.yscale('log')
    plt.title("SNR(i) (Log Scale)")
    plt.xlabel("Timestep i")
    plt.ylabel("SNR")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def verify_teacher_consistency(diffusion, x0, i=500):
    x0 = x0.to(diffusion.device)
    eps = torch.randn_like(x0)
    t = torch.full((1,), i, device=diffusion.device, dtype=torch.long)

    xi = diffusion.q_sample(x0, t, eps)

    mu_tilde, _ = diffusion.q_posterior_mean_var(x0, xi, i)

    print(f"Teacher Consistency Check at i={i}")
    print(f"xi mean: {xi.mean().item():.4f}")
    print(f"mu_tilde mean: {mu_tilde.mean().item():.4f}")

"""### Task 3"""

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A standard block with GroupNorm, SiLU, and timestep embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x, t_emb):

        h = self.act1(self.norm1(self.conv1(x)))

        time_bias = self.mlp(t_emb)[..., None, None]
        h = h + time_bias

        h = self.act2(self.norm2(self.conv2(h)))
        return h

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = Block(base_channels, base_channels, time_emb_dim)

        self.down2 = Block(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = Block(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_block1 = Block(base_channels * 4, base_channels, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_block2 = Block(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):

        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        d1 = self.down1(x, t_emb)
        p1 = self.pool(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool(d2)

        b = self.bottleneck(p2, t_emb)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_block2(u2, t_emb)

        return self.final_conv(u2)

"""## Task 4

"""

def train_one_epoch(model, loader, diffusion, optimizer, device):
    model.train()
    epoch_losses = []
    grad_norms = []

    for x0, _ in loader:
        x0 = x0.to(device)
        t = diffusion.sample_timesteps(x0.shape[0])
        noise = torch.randn_like(x0)

        xi = diffusion.q_sample(x0, t, noise)
        pred_noise = model(xi, t)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()

        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        epoch_losses.append(loss.item())
        grad_norms.append(total_grad_norm)

    return epoch_losses, grad_norms

@torch.no_grad()
def save_periodic_grid(model, diffusion, epoch, folder="samples"):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    torch.manual_seed(42)
    samples, _, _ = ddpm_sampler(model, diffusion, batch_size=64)
    grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(grid, f"{folder}/epoch_{epoch:03d}.png")

"""## Task 5"""

@torch.no_grad()
def ddpm_sampler(model, diffusion, batch_size=64):
    model.eval()
    device = diffusion.device
    x = torch.randn((batch_size, 1, 28, 28), device=device)
    intermediate_x = {}
    norms = []
    targets = {diffusion.L, (3 * diffusion.L) // 4, diffusion.L // 2, diffusion.L // 4, 1}
    for i in range(diffusion.L, 0, -1):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        norms.append(torch.norm(x).item())
        if i in targets:
            intermediate_x[i] = x.cpu()
        eps_hat     = model(x, t)
        alpha_i     = diffusion.alphas[i - 1].view(-1, 1, 1, 1)
        beta_i      = diffusion.betas[i - 1].view(-1, 1, 1, 1)
        alpha_bar_i = diffusion.alpha_bars[i - 1].view(-1, 1, 1, 1)
        mean = (1 / torch.sqrt(alpha_i)) * (x - (beta_i / torch.sqrt(1 - alpha_bar_i)) * eps_hat)
        if i > 1:
            z   = torch.randn_like(x)
            var = diffusion.posterior_variance[i - 1].view(-1, 1, 1, 1)
            x   = mean + torch.sqrt(var) * z
        else:
            x = mean
    return x, intermediate_x, norms

"""## Task 6

"""

def make_cosine_schedule(L, s=0.008, device='cuda'):

    steps = L + 1
    t = torch.linspace(0, L, steps, dtype=torch.float64, device=device)

    f_t = torch.cos(((t / L) + s) / (1 + s) * (math.pi / 2)) ** 2

    alphas_bar = f_t / f_t[0]

    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])

    return torch.clip(betas, min=0.0001, max=0.999).float()

@torch.no_grad()
def ddim_sample(model, diffusion, batch_size=64):
    model.eval()
    device = diffusion.device
    x = torch.randn((batch_size, 1, 28, 28), device=device)

    for i in range(diffusion.L, 0, -1):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        eps_hat          = model(x, t)
        alpha_bar_t      = diffusion.alpha_bars[i - 1].view(-1, 1, 1, 1)
        alpha_bar_t_prev = diffusion.alpha_bars_prev[i - 1].view(-1, 1, 1, 1)

        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
        x       = torch.sqrt(alpha_bar_t_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_t_prev) * eps_hat

    return x

"""## Task 7"""

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2   = nn.MaxPool2d(2)
        self.fc      = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x, return_features=False):
        h = torch.relu(self.conv1(x))
        h = self.pool1(h)
        h = torch.relu(self.conv2(h))
        h = self.pool2(h)
        h = h.view(h.size(0), -1)
        if return_features:
            return h
        return self.fc(h)

@torch.no_grad()
def compute_fid_kid(classifier, real_images, gen_images):
    phi_real = classifier(real_images, return_features=True).cpu().numpy()
    phi_gen  = classifier(gen_images,  return_features=True).cpu().numpy()

    mu_r, mu_g       = phi_real.mean(0), phi_gen.mean(0)
    sigma_r, sigma_g = np.cov(phi_real, rowvar=False), np.cov(phi_gen, rowvar=False)

    diff     = mu_r - mu_g
    covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2.0 * covmean)

    d    = phi_real.shape[1]
    K_rr = (np.dot(phi_real, phi_real.T) / d + 1) ** 3
    K_gg = (np.dot(phi_gen,  phi_gen.T)  / d + 1) ** 3
    K_rg = (np.dot(phi_real, phi_gen.T)  / d + 1) ** 3
    m, n = phi_real.shape[0], phi_gen.shape[0]
    kid  = (np.sum(K_rr) - np.trace(K_rr)) / (m * (m - 1)) + (np.sum(K_gg) - np.trace(K_gg)) / (n * (n - 1)) - 2 * np.mean(K_rg)

    return fid, kid

@torch.no_grad()
def check_memorization(gen_batch, train_dataset_tensors):

    B = gen_batch.size(0)

    gen_flat = gen_batch.view(B, -1)
    train_flat = train_dataset_tensors.view(train_dataset_tensors.size(0), -1)

    distances = torch.cdist(gen_flat, train_flat, p=2.0)

    min_dists, nearest_indices = torch.min(distances, dim=1)

    nearest_train_images = train_dataset_tensors[nearest_indices]

    return gen_batch, nearest_train_images, min_dists

@torch.no_grad()
def compute_bpd(model, diffusion, x0):
    B = x0.size(0)
    D = x0.numel() / B
    elbo = torch.zeros(B, device=diffusion.device)

    alpha_bar_L = diffusion.alpha_bars[-1]
    mu_q_L  = torch.sqrt(alpha_bar_L) * x0
    var_q_L = 1.0 - alpha_bar_L
    kl_prior = 0.5 * (var_q_L + mu_q_L**2 - 1 - torch.log(var_q_L))
    elbo += kl_prior.view(B, -1).sum(dim=1)

    for i in range(2, diffusion.L + 1):
        t     = torch.full((B,), i, device=diffusion.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xi    = diffusion.q_sample(x0, t, noise)

        mu_q, var_q = diffusion.q_posterior_mean_var(x0, xi, i)

        eps_hat = model(xi, t)
        alpha_i = diffusion.alphas[i - 1].view(-1, 1, 1, 1)
        beta_i  = diffusion.betas[i - 1].view(-1, 1, 1, 1)
        alpha_bar_i = diffusion.alpha_bars[i - 1].view(-1, 1, 1, 1)

        mu_p  = (1 / torch.sqrt(alpha_i)) * (xi - (beta_i / torch.sqrt(1 - alpha_bar_i)) * eps_hat)
        var_p = diffusion.posterior_variance[i - 1].view(-1, 1, 1, 1)

        kl_i  = 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1)
        elbo += kl_i.view(B, -1).sum(dim=1)

    t1      = torch.full((B,), 1, device=diffusion.device, dtype=torch.long)
    noise1  = torch.randn_like(x0)
    x1      = diffusion.q_sample(x0, t1, noise1)
    eps_hat1 = model(x1, t1)

    alpha_1     = diffusion.alphas[0].view(-1, 1, 1, 1)
    beta_1      = diffusion.betas[0].view(-1, 1, 1, 1)
    alpha_bar_1 = diffusion.alpha_bars[0].view(-1, 1, 1, 1)

    mu_p1 = (1 / torch.sqrt(alpha_1)) * (x1 - (beta_1 / torch.sqrt(1 - alpha_bar_1)) * eps_hat1)

    recon_loss = 0.5 * ((x0 - mu_p1)**2).view(B, -1).sum(dim=1)
    elbo += recon_loss

    bpd = elbo / (D * math.log(2))
    return bpd.mean().item()

os.makedirs("outputs", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x0_sample = train_dataset[0][0].unsqueeze(0).to(device)
print(f"Image shape: {x0_sample.shape}  range: [{x0_sample.min():.2f}, {x0_sample.max():.2f}]")

# Tasks 1 and 2
diffusion = DDPM(L=1000, beta_min=1e-4, beta_max=0.02, device=device)

verify_schedule(diffusion)

verify_teacher_consistency(diffusion, x0_sample)

# Task 3

model = UNet(in_channels=1, base_channels=32).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"UNet parameters: {n_params:,}")

test_t = torch.randint(1, 1001, (x0_sample.shape[0],), device=device)
test_out = model(x0_sample, test_t)
print(f"UNet output shape: {test_out.shape}")

# Task 4

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

all_losses, all_grad_norms = [], []
num_epochs = 20

for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
    epoch_losses, epoch_grad_norms = train_one_epoch(model, train_loader, diffusion, optimizer, device)
    all_losses.extend(epoch_losses)
    all_grad_norms.extend(epoch_grad_norms)

    if epoch % 5 == 0:
        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch}/{num_epochs}  avg_loss={avg:.4f}  grad_norm={epoch_grad_norms[-1]:.4f}")
        save_periodic_grid(model, diffusion, epoch)

torch.save(model.state_dict(), "outputs/model.pt")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
window = max(1, len(all_losses) // 100)
smoothed = np.convolve(all_losses, np.ones(window) / window, mode='valid')
plt.plot(all_losses, alpha=0.2, label='raw')
plt.plot(smoothed, label='smoothed')
plt.xlabel("Step"); plt.ylabel("MSE loss"); plt.title("Training Loss"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(all_grad_norms, alpha=0.5)
plt.xlabel("Step"); plt.ylabel("Grad norm"); plt.title("Gradient Norms")
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=120)
plt.show()

# Task 5

torch.manual_seed(42)
samples, intermediate_x, norms = ddpm_sampler(model, diffusion, batch_size=64)
samples = samples.cpu()

grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(grid, "outputs/samples_final.png")
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis('off'); plt.title("Generated Samples"); plt.show()

trajectory_imgs = [intermediate_x[k] for k in sorted(intermediate_x.keys(), reverse=True)]
traj_first = [imgs[0] for imgs in trajectory_imgs]
traj_grid = torchvision.utils.make_grid(traj_first, nrow=len(traj_first), normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(traj_grid, "outputs/trajectory.png")
plt.figure(figsize=(12, 2))
plt.imshow(traj_grid.permute(1, 2, 0).numpy())
plt.axis('off')
plt.title(f"Denoising trajectory at steps {sorted(intermediate_x.keys(), reverse=True)}")
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(norms)
plt.xlabel("Reverse step (L to 1)"); plt.ylabel("||x_i||"); plt.title("Sample norm over the reverse process")
plt.tight_layout(); plt.savefig("outputs/sample_norms.png", dpi=120); plt.show()

# Task 6

# cosine schedule
cosine_betas = make_cosine_schedule(L=1000, device=device)
diffusion_cosine = DDPM.__new__(DDPM)
diffusion_cosine.L = 1000
diffusion_cosine.device = device
diffusion_cosine.betas = cosine_betas
diffusion_cosine.alphas = 1.0 - cosine_betas
diffusion_cosine.alpha_bars = torch.cumprod(diffusion_cosine.alphas, dim=0)
diffusion_cosine.alpha_bars_prev = F.pad(diffusion_cosine.alpha_bars[:-1], (1, 0), value=1.0)
diffusion_cosine.sqrt_alpha_bars = torch.sqrt(diffusion_cosine.alpha_bars)
diffusion_cosine.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - diffusion_cosine.alpha_bars)
diffusion_cosine.posterior_variance = (
    diffusion_cosine.betas * (1.0 - diffusion_cosine.alpha_bars_prev) / (1.0 - diffusion_cosine.alpha_bars)
)

ab_linear = diffusion.alpha_bars.cpu().numpy()
ab_cosine = diffusion_cosine.alpha_bars.cpu().numpy()
steps = np.arange(1, 1001)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, ab_linear, label='linear')
plt.plot(steps, ab_cosine, label='cosine')
plt.title(r"$\bar{{\alpha}}_i$ comparison"); plt.xlabel("i"); plt.legend()
plt.subplot(1, 2, 2)
plt.semilogy(steps, ab_linear / (1 - ab_linear), label='linear')
plt.semilogy(steps, ab_cosine / (1 - ab_cosine), label='cosine')
plt.title("SNR(i) comparison"); plt.xlabel("i"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/schedule_ablation.png", dpi=120); plt.show()

# ddim sampling

torch.manual_seed(42)
ddim_samples = ddim_sample(model, diffusion, batch_size=64).cpu()
ddim_grid = torchvision.utils.make_grid(ddim_samples, nrow=8, normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(ddim_grid, "outputs/samples_ddim.png")
plt.figure(figsize=(10, 10))
plt.imshow(ddim_grid.permute(1, 2, 0).numpy())
plt.axis('off'); plt.title("DDIM Samples (deterministic)"); plt.show()

# Task 7
classifier = MNISTClassifier().to(device)
cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

classifier.train()
for epoch in range(5):
    for x0, labels in train_loader:
        x0, labels = x0.to(device), labels.to(device)
        loss = F.cross_entropy(classifier(x0), labels)
        cls_optimizer.zero_grad(); loss.backward(); cls_optimizer.step()
print(f"Extractor trained.")
classifier.eval()

# fid and kid
real_batch = next(iter(train_loader))[0][:512].to(device)
gen_batch_eval, _, _ = ddpm_sampler(model, diffusion, batch_size=512)
gen_batch_eval = gen_batch_eval.to(device)

fid, kid = compute_fid_kid(classifier, real_batch, gen_batch_eval)
print(f"Dataset-FID : {fid:.4f}")
print(f"Dataset-KID : {kid:.6f}")

# checking for memorization
train_tensors = torch.stack([train_dataset[i][0] for i in range(2000)])
gen_check, nearest, min_dists = check_memorization(samples[:16].cpu(), train_tensors)

pairs = []
for g, n in zip(gen_check, nearest):
    pairs.extend([g, n])
pair_grid = torchvision.utils.make_grid(pairs, nrow=8, normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(pair_grid, "outputs/memorization_check.png")
plt.figure(figsize=(10, 4))
plt.imshow(pair_grid.permute(1, 2, 0).numpy())
plt.axis('off'); plt.title("Generated (odd cols) vs Nearest Training Image (even cols)"); plt.show()
print(f"Mean L2 distance to nearest training image: {min_dists.mean():.4f}")

# finding the bpd
test_batch = next(iter(train_loader))[0][:16].to(device)
bpd = compute_bpd(model, diffusion, test_batch)
print(f"Estimated BPD: {bpd:.4f}")
