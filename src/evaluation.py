import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch.nn.functional import softmax
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from train2 import CCF
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#region logging stuff
# Set up logging to file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('myapp.log')
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Common format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already added (avoid duplicate logs on rerun)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
#endregion

def get_accuracy(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # Get classification logits explicitly
            logits = model.classify(x)  
            # Compute predicted class
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def compute_nll(model, dataloader, device='cpu'):
    model.to(device)
    nll = 0.0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model.classify(x)  
            log_probs = F.log_softmax(logits, dim=1)
            nll += F.nll_loss(log_probs, y, reduction='sum').item()
            total += y.size(0)
    return nll / total

def generate_samples(model, num_samples, steps, step_size, noise):
    model = model.to(device)
    model.eval()
    xt = torch.rand((num_samples, 1, 28, 28), device=device) * 2 - 1  # Init in [-1, 1]

    for t in range(steps):
        xt.requires_grad_(True)

        # Proper energy function for JEM
        logits = model.classify(xt)
        energy = -torch.logsumexp(logits, dim=1).sum()

        # Compute gradients
        grad = torch.autograd.grad(energy, xt)[0]

        # Gradient clipping
        grad_norm = grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
        too_large = grad_norm > 10.0
        scale = torch.ones_like(grad_norm)
        scale[too_large] = 10.0 / grad_norm[too_large]
        grad = grad * scale

        # SGLD update
        xt = xt - step_size * grad + noise * torch.randn_like(xt)
        xt = xt.clamp(-1.0, 1.0)
        xt = xt.detach()  # Detach so we don’t backprop through history

    return xt.cpu()

def save_grid(images, path, title):
    grid = make_grid(images, nrow=5, normalize=True, pad_value=1)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig(path)
    plt.close()
    
def plot_energy_histogram(model, real_images, fake_images, path):
    with torch.no_grad():
        real_logits = model.classify(real_images.to(device))
        fake_logits = model.classify(fake_images.to(device))

        real_energy = -torch.logsumexp(real_logits, dim=1).cpu().numpy()
        fake_energy = -torch.logsumexp(fake_logits, dim=1).cpu().numpy()

        plt.figure(figsize=(6, 4))
        sns.histplot(real_energy, label='Real Images', color='green', kde=True)
        sns.histplot(fake_energy, label='Generated Images', color='red', kde=True)
        plt.title('Energy Distribution')
        plt.xlabel('Energy')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def plot_generated_class_distribution(model, samples, path):
    with torch.no_grad():
        logits = model.classify(samples.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        plt.figure(figsize=(6, 4))
        sns.countplot(x=preds)
        plt.title('Class Distribution of Generated Samples')
        plt.xlabel('Predicted Digit')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
if __name__ == "__main__":
    # MNIST test set
    test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ]))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    out_dim = 10
    SHAPE = (1, 28, 28)
    jem_model = CCF()
    base_model = CCF()
    
    # Load model weights and move to device
    jem_model.load_state_dict(torch.load('evaluation_models/jem/jem_model_final.pth', map_location=device))    # wobbly-aardvark-25
    jem_model.to(device)
    
    base_model.load_state_dict(torch.load('evaluation_models/jem0/jem0_model_final.pth', map_location=device)) # true-surf-24
    base_model.to(device)

    # ------------------------------------------------------------------------------------------------------------------------
    # Evaluate accuracy
    # ------------------------------------------------------------------------------------------------------------------------
    base_accuracy = get_accuracy(base_model, test_loader)
    logger.info(f"Accuracy of pretrained base model: {base_accuracy}")
    
    jem_accuracy = get_accuracy(jem_model, test_loader)
    logger.info(f"Accuracy of JEM model: {jem_accuracy}")
    
    # Add this to csv
    with open('evaluation_results.csv', 'a') as f:
        f.write(f"Base Acc: {base_accuracy}, JEM Acc: {jem_accuracy}\n")
    # ------------------------------------------------------------------------------------------------------------------------
    # Evaluate NLL
    # ------------------------------------------------------------------------------------------------------------------------
    base_nll = compute_nll(base_model, test_loader)
    logger.info(f"NLL of pretrained base model: {base_nll}")
    
    jem_nll = compute_nll(jem_model, test_loader)
    logger.info(f"NLL of JEM model: {jem_nll}")
    
    # Add this to csv
    with open('evaluation_results.csv', 'a') as f:
        f.write(f"Base NLL: {base_nll}, JEM NLL: {jem_nll}\n")
    # ------------------------------------------------------------------------------------------------------------------------
    # Generate samples
    # ------------------------------------------------------------------------------------------------------------------------
    # === Parameters for sampling ===
    sample_steps = 70       # Number of SGLD steps
    step_size = 0.5        # SGLD step size
    noise_scale = 0.005     # Noise in SGLD
    num_samples = 20        # Total samples to generate

    samples_jem = generate_samples(jem_model, num_samples, sample_steps, step_size, noise_scale)
    samples_jem0 = generate_samples(base_model, num_samples, sample_steps, step_size, noise_scale)
    
    os.makedirs('evaluation_images/comparison', exist_ok=True)
    save_grid(samples_jem, 'evaluation_images/comparison/jem_samples.png', 'JEM Samples')
    save_grid(samples_jem0, 'evaluation_images/comparison/jem0_samples.png', 'JEM0 Samples')
    logger.info("Sample generation complete. Check 'images/comparison' for generated images.")
    
    # Use 500 real samples
    real_batch = next(iter(test_loader))[0][:500].to(device)

    # Plot energy histograms
    plot_energy_histogram(jem_model, real_batch, samples_jem, 'evaluation_images/comparison/energy_hist_jem.png')
    plot_energy_histogram(base_model, real_batch, samples_jem0, 'evaluation_images/comparison/energy_hist_jem0.png')

    # Plot predicted class distribution
    plot_generated_class_distribution(jem_model, samples_jem, 'evaluation_images/comparison/gen_class_dist_jem.png')
    plot_generated_class_distribution(base_model, samples_jem0, 'evaluation_images/comparison/gen_class_dist_jem0.png')

    logger.info("Diagnostics complete. Check 'evaluation_images/comparison' for visualizations.")
    
    # ------------------------------------------------------------------------------------------------------------------------
