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

verbose_interval = 50

SHAPE = (1, 28, 28)

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
          checkpoint_interval: int,
          do_revert: bool = False,
          model_dir: str = './models',
          image_dir: str | None = None):

    if image_dir is not None:
        os.makedirs(image_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize buffer filled with random samples
    buffer = torch.rand(buffer_size, *SHAPE) * 2 - 1

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_epochs, gamma=learning_rate_decay)
    criterion_clf = torch.nn.CrossEntropyLoss(reduction='mean')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

            # perform SGLD
            model.eval() # Disables dropout
            for t in range(steps):
                xt = xt.clone().detach().requires_grad_(True).to(device)
                xt.retain_grad()
                xt_logits = model(xt)
                logsumexp = torch.logsumexp(xt_logits, dim=-1)
                
                # compute gradients
                logsumexp.sum().backward()
                grad = xt.grad

                # do step manually
                with torch.no_grad():
                    xt = xt + step_size * grad + noise * torch.randn_like(xt)
            model.train()

            # get generation loss            
            xt_logits = model(xt.to(device))
            loss_gen = torch.logsumexp(xt_logits, dim=-1) - torch.logsumexp(x_logits, dim=-1)
            loss = loss_clf + gen_weight * loss_gen.mean()

            # If the loss exploded, clear buffer, go back to start of epoch, and reset optimizer
            if loss.abs() > 1e2:
                if not do_revert:
                    return
                print(f'Loss exploded, reverting to last checkpoint')
                model.load_state_dict(torch.load(f'{model_dir}/checkpoint.pth', weights_only=True))
                buffer = torch.rand(buffer_size, *SHAPE) * 2 - 1
                e = checkpoint_epoch
                b = checkpoint_batch
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                failed = True
                break

            # do model step using optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

            if verbose_interval is not None and b % verbose_interval == 0 or loss.item() < -50:
                fig, axs = plt.subplots(2, 1)
                fig.suptitle(f'Epoch {e+1}, Batch {b}')
                img_real = x[0].detach().cpu().numpy().reshape(SHAPE).transpose(1, 2, 0)
                axs[0].imshow(img_real * 0.5 + 0.5)
                axs[0].set_title(f'True Y: {y[0].detach().cpu().numpy()}, predicted Y: {torch.argmax(x_logits[0], dim=-1).detach().cpu().numpy()}, energy: {energy_real:.2f}')
                img_fake = xt[0].detach().cpu().numpy().reshape(SHAPE).transpose(1, 2, 0).clip(-1, 1)
                axs[1].imshow(img_fake * 0.5 + 0.5)
                axs[1].set_title(f'energy: {energy_fake:.2f}')
                axs[0].set_axis_off()
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
        scheduler.step()
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
            'epoch': e + 1
        })        
       
        # Save checkpoint
        if not failed and e % checkpoint_interval == 0 and loss.abs() < 10:
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
        x = F.dropout(x, p=0.5, training=self.training)
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

    # full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * 2 - 1)
    # ]))

    out_dim = 10
    model = CNN(out_dim) 
    # model = WideResNet(out_dim)

    cfg = dict(
        step_size = 0.5, 
        noise = 0.01, 
        buffer_size = 10000,
        steps = 50,
        reinit_freq = 0.05,
        epochs = 150,
        batch_size = 64,
        learning_rate = 1e-4,
        gen_weight = 1,
        learning_rate_decay = 0.3,
        learning_rate_epochs = 50,
        checkpoint_interval = 5,
        do_revert = False,
        data_fraction = 0.1,
        train_fraction = 0.8,
    )
        
    # Use 10% of the dataset
    subset_size = int(len(full_dataset) * cfg.pop('data_fraction'))
    indices = np.random.RandomState(seed=42).permutation(len(full_dataset))[:subset_size]
    train_split = int(len(indices) * cfg.pop('train_fraction'))
    train_indices = indices[:train_split]
    test_indices = indices[train_split:]    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    wandb.init(project="generative-modelling", config=cfg)

    train(model = model,
          train_dataset = train_dataset,
          test_dataset = test_dataset,
          image_dir = './images',
          **cfg)
    

        
