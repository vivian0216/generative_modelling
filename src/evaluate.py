import torch
import numpy as np
import torch.nn as nn

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision import models
from sklearn.metrics import accuracy_score
from train2 import CNN, WideResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def evaluate(model, x, y):
    """
    Evaluate the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        x: Input tensor of shape (N, 1, 28, 28).
        y: Label tensor of shape (N,).

    Returns:
        float: The accuracy of the model.
    """
    y_pred = predict(model, x, device)
    y_pred = y_pred.numpy()
    y_true = y.cpu().numpy()
    score = accuracy_score(y_true, y_pred)
    
    return score
    
    
if __name__ == "__main__":
    full_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ]))
    
    X = torch.stack([full_dataset[i][0] for i in range(len(full_dataset))])
    Y = torch.tensor([full_dataset[i][1] for i in range(len(full_dataset))])
    
    out_dim = 10
    SHAPE = (1, 28, 28)
    jem_model = CNN(out_dim)
    wide_model = WideResNet(out_dim)
    
    # We load the models from the saved state dictionaries
    jem_model.load_state_dict(torch.load('models/checkpoint.pth', map_location=device))
    wide_model.load_state_dict(torch.load('models/wide_mnist.pth', map_location=device))
