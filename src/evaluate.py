import torch
import numpy as np
import torch.nn as nn
import logging
import os
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import accuracy_score
from train2 import CNN, WideResNet
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


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

def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Predict labels using the given model.

    Args:
        model: Trained model.
        x: Input tensor of shape (N, 1, 28, 28).

    Returns:
        Predicted labels (tensor of shape (N,))
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
    return preds.cpu()


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def generate_samples(model, num_samples=10000, steps=60, step_size=0.01, noise=0.005):
    model.eval()
    samples = []

    total_batches = (num_samples + 63) // 64  # batch size = 64

    for batch_idx in tqdm(range(total_batches), desc="Generating batches"):
        xt = torch.rand(64, 1, 28, 28).to(device) * 2 - 1  # In range [-1, 1]

        for step in range(steps):
            xt.requires_grad_(True)
            logits = model(xt)
            logsumexp = torch.logsumexp(logits, dim=-1)
            grad = torch.autograd.grad(logsumexp.sum(), xt)[0]
            xt = xt + step_size * grad + noise * torch.randn_like(xt)
            xt = xt.clamp(-1, 1).detach()

        samples.append(xt.cpu())

    return torch.cat(samples)[:num_samples]

def convert_to_rgb_and_resize(samples, size=299):
    """Convert grayscale samples to RGB and resize"""
    # 1. Convert from [-1,1] to [0,1]
    samples = (samples + 1) / 2
    
    # 2. Create RGB by repeating the channel 3 times
    rgb_samples = samples.repeat(1, 3, 1, 1)
    
    # 3. Resize to target size
    resize_transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    
    resized_samples = torch.stack([resize_transform(img) for img in rgb_samples])
    
    # 4. Convert to uint8 (0-255) - THIS IS CRUCIAL FOR FID
    resized_samples = (resized_samples * 255).type(torch.uint8)
    
    return resized_samples
def save_images(samples, folder="generated_images", num_images=25, color=False):
    os.makedirs(folder, exist_ok=True)

    samples = samples[:num_images]
    samples = (samples + 1) / 2  # Scale from [-1,1] to [0,1]

    for i, img in enumerate(samples):
        if color:
            img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        else:
            img = img.squeeze().numpy()
        
        plt.imshow(img, cmap=None if color else "gray")
        plt.axis("off")
        plt.savefig(os.path.join(folder, f"sample_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    
if __name__ == "__main__":
    full_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ]))

    # Create DataLoader
    dataloader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    out_dim = 10
    SHAPE = (1, 28, 28)
    jem_model = CNN(out_dim)
    base_model = CNN(out_dim)
    
    # Load model weights and move to device
    jem_model.load_state_dict(torch.load('models/jem/mnist-run-3.pth', map_location=device))
    jem_model.to(device)
    
    base_model.load_state_dict(torch.load('models/cnn/model.pth', map_location=device))
    base_model.to(device)

    # ------------------------------------------------------------------------------------------------------------------------
    # Evaluate accuracy
    # ------------------------------------------------------------------------------------------------------------------------
    accuracy = evaluate_model(base_model, dataloader)
    logger.info(f"Accuracy of pretrained base model: {accuracy}")
    
    #------------------------------------------------------------------------------------------------------------------------
    # FID Calculation
    #------------------------------------------------------------------------------------------------------------------------
    # Generate samples
    # fake_raw = generate_samples(jem_model, num_samples=10000)
    
    # # Convert to RGB and resize
    # fake_images = convert_to_rgb_and_resize(fake_raw)
    # logger.info(f"Generated {len(fake_images)} samples with shape {fake_images.shape}.")
    
    # # Save some samples for visualization
    # save_images(fake_images, folder="generated_images_rgb", color=True)
    # logger.info(f"Generated {len(fake_images)} samples and saved to 'generated_images_rgb' folder.")
    
    # # Prepare real images (already transformed to 3-channel during dataset loading)
    # transform_fid = transforms.Compose([
    #     transforms.Resize(299),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),  # This automatically converts to [0,1] and float32
    #     transforms.Lambda(lambda x: (x * 255).type(torch.uint8))  # Convert to uint8
    # ])
    
    # real_dataset = MNIST(root='./data', train=True, download=True, transform=transform_fid)
    # real_loader = DataLoader(real_dataset, batch_size=64)
    
    # # FID calculation
    # fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # # Real images
    # for batch in real_loader:
    #     imgs = batch[0].to(device)
    #     fid.update(imgs, real=True)

    # # Fake images
    # fake_loader = DataLoader(fake_images, batch_size=64)
    # for batch in fake_loader:
    #     fid.update(batch.to(device), real=False)

    # fid_score = fid.compute()
    # logger.info(f"FID score: {fid_score.item():.2f}")