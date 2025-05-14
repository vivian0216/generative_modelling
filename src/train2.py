import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random 
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

verbose_interval = None

def train(model: nn.Module,
          train_dataset: torch.utils.data.Dataset,
          test_dataset: torch.utils.data.Dataset,
          step_size: float, 
          noise: float, 
          buffer_size: int,
          steps: int,
          reinit_freq: float,
          epochs: int,
          batch_size: int,
          learning_rate: float,
          gen_weight: float,
          learning_rate_decay: float,
          learning_rate_epochs: int,
          model_dir: str = './models'):
    
    buffer = []
    
    # Set up interactive plotting if verbose mode is enabled
    if verbose_interval is not None:
        plt.ion()  # Turn on interactive mode
        plt.rcParams['figure.constrained_layout.use'] = True
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        plt.show(block=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_epochs, gamma=learning_rate_decay)
    criterion_clf = torch.nn.CrossEntropyLoss(reduction='mean')
    
    os.makedirs(model_dir, exist_ok=True)
    model.to(device)
    
    # Track best model to prevent continuous crashing
    best_loss = float('inf')
    patience = 0
    max_patience = 3
    explosion_count = 0
    max_explosions = 5
    
    e = 0 
    while e < epochs:
        # Create a new dataloader with a different seed each epoch to avoid the same batch order
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            # Use epoch number and explosion count to ensure different shuffling after resets
            generator=torch.Generator().manual_seed(42 + e + explosion_count * 100)
        )
        
        # Save checkpoint at the beginning of each epoch
        model_path = f'{model_dir}/checkpoint.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Saved checkpoint to {os.path.abspath(model_path)}')
        
        clf_losses = []
        gen_losses = []
        combined_losses = []
        accs = []
        energies_real = []
        energies_fake = []        
        pbar = tqdm(train_dataloader, unit='batch', desc=f'Epoch {e+1}')
        
        verbose_counter = 0
        explosion_this_epoch = False

        # Train
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # get classification loss
            x_logits = model(x)
            loss_clf = criterion_clf(x_logits, y)

            # sample starting x from buffer
            xt = torch.empty_like(x)
            for i in range(x.shape[0]):
                if len(buffer) == 0 or np.random.random() < reinit_freq:
                    # get a new sample in range [-1, 1)
                    xt[i] = torch.rand(x.shape[1:]) * 2 - 1
                else:
                    # Sample from buffer more carefully
                    buffer_batch = random.choice(buffer)
                    idx = np.random.randint(buffer_batch.shape[0])
                    xt[i] = buffer_batch[idx].clone()

            # perform SGLD with gradient clipping to prevent explosions
            for t in range(steps):
                xt = xt.clone().detach().requires_grad_(True).to(device)
                xt_logits = model(xt)
                logsumexp = torch.logsumexp(xt_logits, dim=-1)
                
                # compute gradients
                logsumexp.sum().backward()
                
                # Clip gradients to prevent extreme values - FIX APPLIED HERE
                grad = xt.grad
                
                # Calculate the norm properly across all dimensions except batch
                # For CIFAR (B, 3, 32, 32) or MNIST (B, 1, 28, 28)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1)
                
                # Reshape gradient norm to match batch dimension for comparison
                grad_norm = grad_norm.view(grad.shape[0], 1, 1, 1)
                
                # Create the mask for values that are too large
                too_large = grad_norm > 10.0
                
                # Apply the mask and rescale gradients
                if too_large.any():
                    # The mask broadcasting will now work correctly
                    scale_factor = 10.0 / grad_norm
                    scale_factor[~too_large] = 1.0
                    grad = grad * scale_factor

                # do step manually with gradient clipping
                with torch.no_grad():
                    xt = xt + step_size * grad + noise * torch.randn_like(xt)
                    # Ensure xt stays in valid range to prevent extreme values
                    xt.clamp_(-1.0, 1.0)

            # get generation loss with safeguards
            xt_logits = model(xt.to(device))
            
            # Apply numerical stability tricks
            x_logsumexp = torch.logsumexp(x_logits, dim=-1)
            xt_logsumexp = torch.logsumexp(xt_logits, dim=-1)
            
            # Clip values to prevent extreme differences
            diff = xt_logsumexp - x_logsumexp
            diff = torch.clamp(diff, -100, 100)
            loss_gen = diff.mean()
            
            # Weighted combined loss with scaled gen_weight
            # Start with smaller gen_weight and increase gradually
            effective_gen_weight = gen_weight * min(1.0, (e + 1) / 10)
            loss = loss_clf + effective_gen_weight * loss_gen

            # If the loss is getting too large, skip this batch
            if not torch.isfinite(loss) or loss.abs() > 1e8:
                print(f'Loss exploded at batch {batch_idx}: {loss.item()}, skipping')
                explosion_this_epoch = True
                explosion_count += 1
                
                if explosion_count >= max_explosions:
                    print(f'Too many explosions, reducing learning rate and gen_weight')
                    learning_rate *= 0.5
                    gen_weight *= 0.5
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    explosion_count = 0
                
                # Load last checkpoint and skip to next batch
                model.load_state_dict(torch.load(model_path, weights_only=True))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                # Skip this batch and continue with the next one
                continue

            # do model step using optimizer
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping to the model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

            # add xt to buffer more safely
            with torch.no_grad():
                # Ensure we're adding properly clamped values
                xt_safe = xt.detach().cpu().clamp(-1.0, 1.0)
                buffer.append(xt_safe)
                if len(buffer) > (buffer_size / batch_size):
                    buffer = buffer[1:]

            # add metrics
            clf_losses.append(loss_clf.item())
            gen_losses.append(loss_gen.item())
            combined_losses.append(loss.item())
            accs.append((torch.argmax(x_logits, dim=-1) == y).detach().cpu().numpy().mean())
            energies_real.append(-x_logsumexp.mean().item())
            energies_fake.append(-xt_logsumexp.mean().item())

            pbar.set_postfix({
                'clf_loss': np.mean(clf_losses[-50:]),  # Show average of last 50 batches
                'gen_loss': np.mean(gen_losses[-50:]),
                'loss': np.mean(combined_losses[-50:]),
                'acc': np.mean(accs[-50:]),
                'lr': optimizer.param_groups[0]['lr']
            })

            verbose_counter += 1
            if verbose_interval is not None and verbose_counter % verbose_interval == 0:
                print(f'\nBatch stats:')
                print(f'True Y: {y[:5].detach().cpu().numpy()}')
                print(f'Pred Y: {torch.argmax(x_logits[:5], dim=-1).detach().cpu().numpy()}')

                print(f'Range of x: {x.min().detach().cpu().numpy():.3f} - {x.max().detach().cpu().numpy():.3f}')
                print(f'Range of xt: {xt.min().detach().cpu().numpy():.3f} - {xt.max().detach().cpu().numpy():.3f}')
                
                print(f'Loss of classification: {loss_clf.item():.4f}')
                print(f'Loss of generation: {loss_gen.mean().item():.4f}')
                print(f'Combined loss: {loss.item():.4f}')

                try:
                    # Clear previous plots
                    for ax in axs:
                        ax.clear()
                    
                    # Handle visualization for both MNIST and CIFAR-10
                    x_np = x[0].detach().cpu().numpy()
                    xt_np = xt[0].detach().cpu().numpy()
                    
                    # Check image dimensions to determine dataset
                    if x_np.shape[0] == 1:  # MNIST (1x28x28)
                        axs[0].imshow(x_np.reshape(28, 28), cmap='gray')
                        axs[1].imshow(xt_np.reshape(28, 28), cmap='gray')
                    elif x_np.shape[0] == 3:  # CIFAR-10 (3x32x32)
                        # Transpose from (C,H,W) to (H,W,C) for plotting and clip to valid range
                        axs[0].imshow(np.clip(np.transpose(x_np, (1, 2, 0)) * 0.5 + 0.5, 0, 1))
                        axs[1].imshow(np.clip(np.transpose(xt_np, (1, 2, 0)) * 0.5 + 0.5, 0, 1))
                    else:
                        print(f"Unexpected image shape: {x_np.shape}")
                    
                    axs[0].set_title(f'True Y: {y[0].detach().cpu().numpy()}, predicted: {torch.argmax(x_logits[0], dim=-1).detach().cpu().numpy()}, energy: {energies_real[-1]:.2f}')
                    axs[1].set_title(f'Generated sample, energy: {energies_fake[-1]:.2f}')
                    
                    fig.canvas.draw_idle()   # Update the figure
                    fig.canvas.flush_events()  # Flush the GUI events
                    
                except Exception as e:
                    print(f"Warning: Failed to visualize samples - {e}")

        # If we had an explosion this epoch, don't increment the epoch counter
        # to effectively restart the epoch
        if explosion_this_epoch:
            print(f"Epoch had explosions, resetting and trying again")
            continue
        
        # Save best model based on training loss
        current_loss = np.mean(combined_losses)
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), f'{model_dir}/best_model.pth')
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"No improvement for {max_patience} epochs, loading best model and reducing learning rate")
            model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', weights_only=True))
            learning_rate *= 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            patience = 0

        scheduler.step()

        # Test
        model.eval()
        with torch.no_grad():
            test_accs = []
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            for x, y in tqdm(test_dataloader, unit='batch', desc=f'Testing Epoch {e+1}'):
                x = x.to(device)
                y = y.to(device)
                x_logits = model(x)
                test_accs.append((torch.argmax(x_logits, dim=-1) == y).detach().cpu().numpy().mean())

        # log metrics
        wandb.log({
            'Training/loss_clf': np.mean(clf_losses),
            'Training/loss_gen': np.mean(gen_losses),
            'Training/loss': np.mean(combined_losses),
            'Training/acc': np.mean(accs),
            'Training/energy_real': np.mean(energies_real),
            'Training/energy_fake': np.mean(energies_fake),
            'Testing/acc': np.mean(test_accs),
            'Hyperparams/learning_rate': optimizer.param_groups[0]['lr'],
            'Hyperparams/gen_weight': effective_gen_weight,
            'epoch': e
        })
        e += 1

    # Save final model
    model_path = f'{model_dir}/model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Saved final model to {os.path.abspath(model_path)}')

class CNN(nn.Module):
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class WideResNet(nn.Module):
    def __init__(self, out_dim):
        super(WideResNet, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)

    def forward(self, x):
        # Handle different input channels for MNIST (1) vs CIFAR (3)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat grayscale channel to make 3 channels
        return self.model(x)

if __name__ == "__main__":

    from torchvision.datasets import MNIST, CIFAR10
    from torchvision import transforms
    from torchvision import models
    import argparse

    parser = argparse.ArgumentParser(description='JEM Training')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='dataset to use (default: mnist)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output with sample visualizations')
    args = parser.parse_args()

    # Set verbose interval if flag is provided
    if args.verbose:
        verbose_interval = 100  # Show samples every 100 batches
    
    if args.dataset == 'mnist':
        print("Using MNIST dataset")
        full_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ]))
        test_full_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ]))
        input_channels = 1
    else:
        print("Using CIFAR10 dataset")
        full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ]))
        test_full_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ]))
        input_channels = 3

    out_dim = 10
    
    # Model selection based on dataset
    if args.dataset == 'mnist':
        model = CNN(out_dim)  # CNN works well for MNIST
    else:
        model = WideResNet(out_dim)  # WideResNet better for CIFAR

    # Configure dataset-specific hyperparameters
    if args.dataset == 'mnist':
        cfg = dict(
            step_size = 0.5,  # Lower step size for MNIST
            noise = 0.01, 
            buffer_size = 1000,
            steps = 20,
            reinit_freq = 0.05,
            epochs = 150,
            batch_size = 50,
            learning_rate = 1e-4,
            gen_weight = 0.5,  # Lower gen_weight for stability
            learning_rate_decay = 0.3,
            learning_rate_epochs = 50,
            data_fraction = 1.0,  # Use full training set
        )
    else:
        cfg = dict(
            step_size = 0.25,  # Lower step size for CIFAR
            noise = 0.005,  # Lower noise for CIFAR
            buffer_size = 1000,
            steps = 20,
            reinit_freq = 0.05,
            epochs = 150,
            batch_size = 40,  # Smaller batch size due to complexity
            learning_rate = 5e-5,  # Lower learning rate for stability
            gen_weight = 0.1,  # Start with much lower gen_weight for stability
            learning_rate_decay = 0.3,
            learning_rate_epochs = 50,
            data_fraction = 0.5,  # Use less data initially for faster iteration
        )
        
    # Use a subset of the dataset if specified
    data_fraction = cfg.pop('data_fraction')
    
    if data_fraction < 1.0:
        # Sample a subset of the training data
        train_size = int(len(full_dataset) * data_fraction)
        indices = np.random.RandomState(seed=42).permutation(len(full_dataset))[:train_size]
        train_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        train_dataset = full_dataset
        
    # Use the full test dataset
    test_dataset = test_full_dataset
    
    wandb.init(project=f"jem-{args.dataset}", config=cfg)

    train(model = model,
          train_dataset = train_dataset,
          test_dataset = test_dataset,
          **cfg)