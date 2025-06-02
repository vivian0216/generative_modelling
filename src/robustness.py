import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd
import foolbox as fb

# Initialize a list to store results
results = []

# Helper function to append results
def record_results(model_name, attack_type, epsilon, accuracy):
    results.append({
        "Model": model_name,
        "Attack": attack_type,
        "Epsilon": epsilon,
        "Adversarial Accuracy (%)": round(accuracy, 2)
    })

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

# Load the JEM-0 model (no sampling)
jem_model = CNN(out_dim=10)
jem_model.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
jem_model.to(device)
jem_model.eval()

# Create foolbox model for baseline
fmodel_baseline = fb.PyTorchModel(baseline_model, bounds=(-1, 1))

# Create foolbox models for JEM variants
fmodel_jem = fb.PyTorchModel(jem_model, bounds=(-1, 1))
fmodel_jem = fb.PyTorchModel(jem_model, bounds=(-1, 1))

# Langevin dynamics sampling function
def jem_sample(model, x, n_steps=10, step_size=0.5, noise_std=0.01):
    x_k = x.clone().detach().requires_grad_(True).to(device)
    
    for _ in range(n_steps):
        outputs = model(x_k)
        log_p_y_given_x = F.log_softmax(outputs, dim=1)
        energy = -log_p_y_given_x.mean()  # minimize -log p(y|x)
        
        # Compute gradients
        grads = torch.autograd.grad(energy, x_k)[0]

        # Langevin update
        x_k = x_k + step_size * grads + noise_std * torch.randn_like(x_k)
        x_k = x_k.detach().requires_grad_(True)  # detach and re-enable gradients

        # Clamp to valid image range [-1, 1]
        x_k = x_k.clamp(-1, 1).detach().requires_grad_(True)

    return x_k.detach()

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

# Get a batch of test images and labels
test_iter = iter(test_loader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)


linf_attack = fb.attacks.LinfPGD()
linf_epsilons = [0.1, 0.2, 0.3, 0.4]  # Typical range for L-inf
l2_attack = fb.attacks.L2PGD()
l2_epsilons = [1.0, 2.0, 3.0, 4.0]  # Typical range for L2

def run_attacks(fmodel, model_name, images, labels, linf_epsilons = [0.1, 0.2, 0.3, 0.4], l2_epsilons = [1.0, 2.0, 3.0, 4.0], sampling_steps=0):
    if sampling_steps > 0:
        x_sampled = jem_sample(fmodel, images, n_steps=sampling_steps)
    else:
        x_sampled = images
    
    # L-infinity PGD attack
    print(f"\nL-infinity PGD Attack Results on {model_name} Model:")
    for eps in linf_epsilons:
        _, adversarial, success = linf_attack(fmodel, x_sampled, labels, epsilons=eps)
        
        with torch.no_grad():
            adv_outputs = fmodel(adversarial)
            _, adv_predicted = torch.max(adv_outputs, 1)
            adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
        record_results(model_name, "Linf", eps, adv_accuracy)
        print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

    # L2 PGD attack
    print(f"\nL2 PGD Attack Results on {model_name} Model:")
    for eps in l2_epsilons:
        _, adversarial, success = l2_attack(fmodel, x_sampled, labels, epsilons=eps)
        
        with torch.no_grad():
            adv_outputs = fmodel(adversarial)
            _, adv_predicted = torch.max(adv_outputs, 1)
            adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
        record_results(model_name, "L2", eps, adv_accuracy)
        print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

run_attacks(fmodel_baseline, "baseline", images, labels)
run_attacks(fmodel_jem, "jem-0", images, labels)
run_attacks(fmodel_jem, "jem-1", images, labels, sampling_steps=1)
run_attacks(fmodel_jem, "jem-10", images, labels, sampling_steps=10)

df = pd.DataFrame(results)

# Save results to CSV
df.to_csv('adversarial_robustness_results.csv', index=False)
print("Results saved to 'adversarial_robustness_results.csv'")