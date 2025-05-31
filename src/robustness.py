import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

# Define the CNN class
class CNN(nn.Module):
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, out_dim)
     
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Load the test dataset
test_dataset = MNIST(root='./data', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: x * 2 - 1)
                     ]))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the baseline CNN model
baseline_model = CNN(out_dim=10)  # MNIST has 10 classes
baseline_model.load_state_dict(torch.load('./models/baseline_model.pth', map_location=device))
baseline_model.to(device)
baseline_model.eval()

# Load the JEM model
jem_model = CNN(out_dim=10)  # MNIST has 10 classes
jem_model.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
jem_model.to(device)
jem_model.eval()

print("Models loaded successfully!")
print(f"Test dataset size: {len(test_dataset)}")

# Create test dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Function to compute accuracy
def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Compute and print accuracies
baseline_accuracy = compute_accuracy(baseline_model, test_loader)
jem_accuracy = compute_accuracy(jem_model, test_loader)

print(f"Baseline model accuracy: {baseline_accuracy:.2f}%")
print(f"JEM model accuracy: {jem_accuracy:.2f}%")

# PGD attack using foolbox
import foolbox as fb

# Create foolbox model for baseline
fmodel_baseline = fb.PyTorchModel(baseline_model, bounds=(-1, 1))

# Get a batch of test images and labels
test_iter = iter(test_loader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)

# L-infinity PGD attack
print("\nL-infinity PGD Attack Results on Baseline Model:")
linf_attack = fb.attacks.LinfPGD()
linf_epsilons = [0.1, 0.2, 0.3, 0.4]  # Typical range for L-inf

for eps in linf_epsilons:
    _, adversarial, success = linf_attack(fmodel_baseline, images, labels, epsilons=eps)
    
    # Calculate accuracy on adversarial examples
    with torch.no_grad():
        adv_outputs = baseline_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_base_Linf = (adv_predicted == labels).float().mean().item() * 100
    
    print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy_base_Linf:.2f}%")

# L2 PGD attack
print("\nL2 PGD Attack Results on Baseline Model:")
l2_attack = fb.attacks.L2PGD()
l2_epsilons = [1.0, 2.0, 3.0, 4.0]  # Typical range for L2

for eps in l2_epsilons:
    _, adversarial, success = l2_attack(fmodel_baseline, images, labels, epsilons=eps)
    
    # Calculate accuracy on adversarial examples
    with torch.no_grad():
        adv_outputs = baseline_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_base_L2 = (adv_predicted == labels).float().mean().item() * 100
    
    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy_base_L2:.2f}%")

# Create foolbox model for JEM
fmodel_jem = fb.PyTorchModel(jem_model, bounds=(-1, 1))

# L-infinity PGD attack on JEM model
print("\nL-infinity PGD Attack Results on JEM Model:")
for eps in linf_epsilons:
    _, adversarial, success = linf_attack(fmodel_jem, images, labels, epsilons=eps)
    
    # Calculate accuracy on adversarial examples
    with torch.no_grad():
        adv_outputs = jem_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_jem_Linf = (adv_predicted == labels).float().mean().item() * 100
    
    print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy_jem_Linf:.2f}%")

# L2 PGD attack on JEM model
print("\nL2 PGD Attack Results on JEM Model:")
for eps in l2_epsilons:
    _, adversarial, success = l2_attack(fmodel_jem, images, labels, epsilons=eps)
    
    # Calculate accuracy on adversarial examples
    with torch.no_grad():
        adv_outputs = jem_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_jem_L2 = (adv_predicted == labels).float().mean().item() * 100
    
    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy_jem_L2:.2f}%")

