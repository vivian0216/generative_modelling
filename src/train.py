import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import inspect
import torch.nn.functional as F
import os
from typing import Callable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SHAPE = (1, 28, 28)

# Default values are the ones used in the paper
def train(model: nn.Module,
          train_dataset: torch.utils.data.Dataset,
          test_dataset: torch.utils.data.Dataset,
          epochs: int = 150,
          batch_size: int = 64,
          optimizer = lambda x: torch.optim.Adam(x, lr=1e-4),
          scheduler = lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=50, gamma=0.3),
          sgld_steps: int = 40, # In the paper they start with 20 and increase to 40. Not explained when they do this, so default to reproduce is use 40 for the whole run.
          sgld_optimizer = lambda x: torch.optim.SGD(x, lr=1.0),
          sgld_scheduler = lambda x: torch.optim.lr_scheduler.ConstantLR(x, factor=1.0),
          sgld_noise: float = 0.01,
          gen_weight: float = 1,
          buffer_size: int = 10000,
          reinit_freq: float = 0.05,         
          loss_explosion_threshold: float | None = 1e8, # If not None, catch when absolute loss is greater than threshold
          revert_on_loss_explosion: bool = False, # If True, revert to last checkpoint if loss explosion is detected
          checkpoint_interval: int = 1,
          verbose_interval: int | None = 50,
          model_dir: str = './models',
          image_dir: str | None = None):

    if image_dir is not None:
        os.makedirs(image_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model.to(device)
    _optimizer = optimizer(model.parameters())
    _scheduler = scheduler(_optimizer)
    criterion_clf = torch.nn.CrossEntropyLoss(reduction='mean')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize buffer filled with random samples
    buffer = torch.rand(buffer_size, *SHAPE) * 2 - 1

    e = 0 
    b = 0
    checkpoint_epoch = 0
    checkpoint_batch = 0
    while e < epochs:      

        # Train
        failed = False
        pbar = tqdm(train_dataloader, unit='batch', desc=f'Epoch {e+1}')
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            # get classification loss
            x_logits = model(x)
            loss_clf = criterion_clf(x_logits, y)

            # sample starting x from buffer, possibly reinitialize
            buffer_indices = np.random.randint(0, buffer_size, size=x.shape[0])
            xt = buffer[buffer_indices]
            for i in range(x.shape[0]):
                if np.random.random() < reinit_freq:
                    xt[i] = torch.rand(x.shape[1:]) * 2 - 1

            # perform SGLD, using torch optimizers            
            xt = xt.detach().to(device).requires_grad_(True)
            _sgld_optimizer = sgld_optimizer([xt])
            _sgld_scheduler = sgld_scheduler(_sgld_optimizer)
            model.eval() # Disables dropout etc.
            for t in range(sgld_steps):
 
                xt_logits = model(xt)
                energy = -torch.logsumexp(xt_logits, dim=-1)
                
                # compute gradients
                _sgld_optimizer.zero_grad()
                energy.sum().backward()
                
                # Do gradient step to minimize the energy using the optimizer
                # Equivalent to maximizing the logsumexp as in the paper's pseudocode
                _sgld_optimizer.step()
                _sgld_scheduler.step()

                # Add noise afterwards to complete the iteration, not decayed by the scheduler
                xt.data += sgld_noise * torch.randn_like(xt)

                # Save images   
                if verbose_interval is not None and b % verbose_interval == 0 or loss.item() < -50 and image_dir is not None:
                    if t % 10 == 0 or sgld_steps - t < 10: # save every 10 steps or the last 10 steps
                        img = xt[0].detach().cpu().numpy().reshape(SHAPE).transpose(1, 2, 0)
                        plt.imshow(img.clip(-1,1) * 0.5 + 0.5)
                        os.makedirs(f'{image_dir}/sgld', exist_ok=True)
                        plt.savefig(f'{image_dir}/sgld/epoch_{e+1}_batch_{b}_step_{t}.png')
                        plt.close()
            model.train()

            # get generation loss            
            xt_logits = model(xt.to(device))
            loss_gen = torch.logsumexp(xt_logits, dim=-1) - torch.logsumexp(x_logits, dim=-1)
            loss = loss_clf + gen_weight * loss_gen.mean()

            # If the loss exploded, clear buffer, go back to start of epoch, and reset optimizer
            if loss_explosion_threshold is not None and loss.abs() > loss_explosion_threshold:
                if not revert_on_loss_explosion:
                    return
                print(f'Loss exploded, reverting to last checkpoint')
                model.load_state_dict(torch.load(f'{model_dir}/checkpoint.pth', weights_only=True))
                buffer = torch.rand(buffer_size, *SHAPE) * 2 - 1
                e = checkpoint_epoch
                b = checkpoint_batch
                _optimizer = optimizer(model.parameters())
                _scheduler = scheduler(_optimizer)
                for _ in range(e):
                    _scheduler.step() # Make sure scheduler is at the correct epoch
                failed = True
                break

            # do model step using optimizer
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            # update buffer with either further optimized or reinitialized samples
            buffer[buffer_indices] = xt.cpu().detach()

            # log metrics
            energy_real = -torch.logsumexp(x_logits, dim=-1).mean().item()
            energy_fake = -torch.logsumexp(xt_logits, dim=-1).mean().item()
            wandb.log({
                'Training/loss_clf': loss_clf.item(),
                'Training/loss_gen': loss_gen.mean().item(),
                'Training/loss': loss.item(),
                'Training/acc':(torch.argmax(x_logits, dim=-1) == y).detach().cpu().numpy().mean(),
                'Training/energy_real': energy_real,
                'Training/energy_fake': energy_fake,
                'Training/max_l1': xt.abs().max().item(),
                'epoch': e + 1,
                'batch': b
            })

            # save images (of first one in batch)
            if verbose_interval is not None and b % verbose_interval == 0:
                fig, axs = plt.subplots(2, 1)
                fig.suptitle(f'Epoch {e+1}, Batch {b}')
                img_real = x[0].detach().cpu().numpy().reshape(SHAPE).transpose(1, 2, 0)
                axs[0].imshow(img_real.clip(-1, 1) * 0.5 + 0.5)
                axs[0].set_title(f'True Y: {y[0].detach().cpu().numpy()}, pred Y: {torch.argmax(x_logits[0], dim=-1).detach().cpu().numpy()}, E={energy_real:.2f}')
                axs[0].set_axis_off()
                img_fake = xt[0].detach().cpu().numpy().reshape(SHAPE).transpose(1, 2, 0)
                axs[1].imshow(img_fake.clip(-1, 1) * 0.5 + 0.5)
                axs[1].set_title(f'SGLD Gen E={energy_fake:.2f}')                
                axs[1].set_axis_off()
                plt.tight_layout()
                if image_dir is not None:
                    plt.savefig(f'{image_dir}/epoch_{e+1}_batch_{b}.png')
                    plt.close()
                else:
                    plt.show()
            b += 1

        if failed:
            continue
        _scheduler.step()
        e += 1

        # Test
        model.eval()
        with torch.no_grad():
            test_accs = []
            for x, y in tqdm(test_dataloader, unit='batch', desc=f'Epoch {e+1}'):
                x = x.to(device)
                y = y.to(device)
                x_logits = model(x)
                test_accs.append((torch.argmax(x_logits, dim=-1) == y).detach().cpu().numpy().mean())
        wandb.log({
            'Testing/acc': np.mean(test_accs),
            'epoch': e
        })

        # Save checkpoint
        if not failed and e % checkpoint_interval == 0:
            model_path = f'{model_dir}/checkpoint.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Saved checkpoint to {os.path.abspath(model_path)}')
            checkpoint_epoch = e
            checkpoint_batch = b
        
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
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class WideResNet(nn.Module):
    def __init__(self, out_dim):
        super(WideResNet, self).__init__()
        self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    from torchvision.datasets import MNIST, CIFAR10
    from torchvision import transforms
    from torchvision import models

    full_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ]))
    out_dim = 10
    SHAPE = (1, 28, 28)
    model = CNN(out_dim)

    # full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * 2 - 1)
    # ]))
    # out_dim = 10
    # SHAPE = (3, 32, 32)
    # model = WideResNet(out_dim)     

    cfg = dict(       
        # Training parameters
        epochs = 150,
        batch_size = 64,
        optimizer = lambda x: torch.optim.Adam(x, lr=1e-4),
        scheduler = lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=50, gamma=0.3),

        # SGLD parameters
        sgld_steps = 40,
        sgld_noise = 0.01, 
        sgld_optimizer = lambda x: torch.optim.SGD(x, lr=1.0),
        sgld_scheduler = lambda x: torch.optim.lr_scheduler.ConstantLR(x, factor=1.0),
        gen_weight = 1,
        buffer_size = 10000,
        reinit_freq = 0.05,

        # Misc
        loss_explosion_threshold = 1e2,
        revert_on_loss_explosion = False,
        checkpoint_interval = 1,
        data_fraction = 1,
        train_fraction = 0.8,
        verbose_interval = 50
    )
        
    # Use 10% of the dataset
    subset_size = int(len(full_dataset) * cfg.pop('data_fraction'))
    indices = np.random.RandomState(seed=42).permutation(len(full_dataset))[:subset_size]
    train_split = int(len(indices) * cfg.pop('train_fraction'))
    train_indices = indices[:train_split]
    test_indices = indices[train_split:]    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Turn the lambdas into strings for logging
    pretty_cfg = cfg.copy()
    for k, v in pretty_cfg.items():
        if isinstance(v, Callable):
            pretty_cfg[k] = inspect.getsource(v).split('lambda x: ')[1][:-2]
    wandb.init(project="generative-modelling", config=pretty_cfg)

    train(model = model,
          train_dataset = train_dataset,
          test_dataset = test_dataset,
          image_dir = './images',
          **cfg)       
