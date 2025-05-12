import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random 
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

verbose_interval = 250

def train(model: nn.Module,
          dataset: torch.utils.data.Dataset,
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
          learning_rate_epochs: list[int]):
    
    buffer = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_clf = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    model.to(device)

    for e in range(epochs):
        clf_losses = []
        gen_losses = []
        combined_losses = []
        accs = []
        energies_real = []
        energies_fake = []        
        pbar = tqdm(dataloader, unit='batch', desc=f'Epoch {e+1}')
        
        verbose_counter = 0

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            # get classification loss
            x_logits = model(x)
            loss_clf = criterion_clf(x_logits, y)

            # # sample starting x from buffer
            if len(buffer) == 0 or np.random.random() < reinit_freq:
                # get a new sample in range [-1, 1)
                xt = torch.rand(x.shape) * 2 - 1
            else:
                xt = random.choice(buffer)

            # perform SGLD
            for t in range(steps):
                xt = xt.clone().detach().requires_grad_(True).to(device)
                xt.retain_grad()
                xt_logits = model(xt)
                logsumexp = torch.logsumexp(xt_logits, dim=-1)
                
                # compute gradients
                logsumexp.backward()
                grad = xt.grad

                # do step manually
                with torch.no_grad():
                    xt = xt + step_size * grad + noise * torch.randn_like(xt)

            # # get generation loss
            xt_logits = model(xt.to(device))
            loss_gen = torch.logsumexp(x_logits, dim=-1) - torch.logsumexp(xt_logits, dim=-1)
            loss = loss_clf + gen_weight * loss_gen

            # do model step using optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add xt to buffer
            buffer.append(xt)
            if len(buffer) > buffer_size:
                buffer = buffer[1:]

            # add metrics
            clf_losses.append(loss_clf.item())
            gen_losses.append(loss_gen.mean().item())
            combined_losses.append(loss.item())
            accs.append((torch.argmax(x_logits, dim=-1) == y).detach().cpu().numpy().mean())
            energies_real.append(-torch.logsumexp(x_logits, dim=-1).mean().item())
            energies_fake.append(-torch.logsumexp(xt_logits, dim=-1).mean().item())

            pbar.set_postfix({
                'clf_loss': np.mean(clf_losses),
                'gen_loss': np.mean(gen_losses),
                'loss': np.mean(combined_losses),
                'acc': np.mean(accs)
            })

            verbose_counter += 1
            if verbose_interval is not None and verbose_counter % verbose_interval == 0:
                print(f'True Y: {y.detach().cpu().numpy()}')
                print(f'Pred Y: {torch.argmax(x_logits, dim=-1).detach().cpu().numpy()}')

                print(f'Range of x: {x.min().detach().cpu().numpy()} - {x.max().detach().cpu().numpy()}')
                print(f'Logits of x: {x_logits.detach().cpu().numpy()}')
                print(f'Energy of x: {-torch.logsumexp(x_logits, dim=-1).detach().cpu().numpy()}')
                
                print(f'Range of xt: {xt.min().detach().cpu().numpy()} - {xt.max().detach().cpu().numpy()}')
                print(f'Logits of xt: {xt_logits.detach().cpu().numpy()}')
                print(f'Energy of xt: {-torch.logsumexp(xt_logits, dim=-1).detach().cpu().numpy()}')

                print(f'Loss of x: {loss_clf.detach().cpu().numpy()}')
                print(f'Loss of xt: {loss_gen.detach().cpu().numpy()}')
                print(f'Loss of combined: {loss.detach().cpu().numpy()}')

                fig, axs = plt.subplots(2, 1)
                axs[0].imshow(x[0].detach().cpu().numpy().reshape(28, 28))
                axs[0].set_title(f'True Y: {y[0].detach().cpu().numpy()}, predicted Y: {torch.argmax(x_logits[0], dim=-1).detach().cpu().numpy()}, energy: {energies_real[-1]}')
                axs[1].imshow(xt[0].detach().cpu().numpy().reshape(28, 28))
                axs[1].set_title(f'energy: {energies_fake[-1]}')
                plt.show()

        # log metrics
        wandb.log({
            'Training/loss_clf': np.mean(clf_losses),
            'Training/loss_gen': np.mean(gen_losses),
            'Training/loss': np.mean(combined_losses),
            'epoch': e
        })


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

if __name__ == "__main__":

    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torchvision import models

    # Get only a fraction of the dataset
    full_dataset = MNIST(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ]))

    out_dim = 10
    model = CNN(out_dim)

    cfg = dict(
        step_size = 1, 
        noise = 0.01, 
        buffer_size = 10000,
        steps = 20,
        reinit_freq = 0.05,
        epochs = 10,
        batch_size = 1,
        learning_rate = 0.001,
        gen_weight = 0,
        learning_rate_decay = None,
        learning_rate_epochs = None,
        train_fraction = 0.05,
    )
        
    # Use 10% of the dataset
    subset_size = int(len(full_dataset) * cfg.pop('train_fraction'))
    indices = np.random.RandomState(seed=42).permutation(len(full_dataset))[:subset_size]
    dataset = torch.utils.data.Subset(full_dataset, indices)
    wandb.init(project="generative-modelling", config=cfg)

    train(model = model,
          dataset = dataset,
          **cfg)
    

        
