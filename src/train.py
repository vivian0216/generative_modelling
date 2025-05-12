import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model: nn.Module,
          dataset: torch.utils.data.Dataset,
          step_size: float, 
          noise: float, 
          buffer_size: int,
          steps: int,
          reinit_freq: float,
          epochs: int,
          learning_rate: float,
          learning_rate_decay: float,
          learning_rate_epochs: list[int]):
    
    buffer = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_clf = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset)

    model.to(device)

    for e in range(epochs):
        for x, y in tqdm(dataloader, unit='batch', desc=f'Epoch {e+1}'):
            x = x.to(device)
            y = y.to(device)
            
            # get classification loss
            x_logits = model(x)
            loss_clf = criterion_clf(x_logits, y)

            # sample starting x from buffer
            if len(buffer) == 0 or np.random.random() < reinit_freq:
                # get a new sample in range [-1, 1)
                xt = torch.rand(x.shape) * 2 - 1
            else:
                xt = random.choice(buffer)

            # perform SGLD
            for t in range(steps):
                xt = torch.tensor(xt.clone().detach(), requires_grad=True).to(device)
                xt.retain_grad()
                xt_logits = model(xt)
                logsumexp = torch.logsumexp(xt_logits, dim=-1)
                
                # compute gradients
                logsumexp.backward()
                grad = xt.grad

                # do step manually
                with torch.no_grad():
                    xt = xt + step_size * grad + noise * torch.randn_like(xt)

            # get generation loss
            xt_logits = model(xt)
            loss_gen = torch.logsumexp(x_logits, dim=-1) - torch.logsumexp(xt_logits, dim=-1)
            loss =  loss_clf + loss_gen

            # do model step using optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add xt to buffer
            buffer.append(xt)
            if len(buffer) > buffer_size:
                buffer = buffer[1:]

        
class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":

    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ]))

    in_dim = 28 * 28
    out_dim = 10

    model = Model(in_dim, out_dim)

    train(model = model,
          dataset = dataset,
          step_size = 1, 
          noise = 0.01, 
          buffer_size = 10000,
          steps = 20,
          reinit_freq = 0.05,
          epochs = 10,
          learning_rate = 1e-2,
          learning_rate_decay = None,
          learning_rate_epochs = None)
    

        
