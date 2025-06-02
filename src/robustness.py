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

class JEMWrapper(nn.Module):
    def __init__(self, base_model, num_steps, step_size, noise_std, eot_samples=1):
        super().__init__()
        self.base = base_model
        self.num_steps = num_steps
        self.step_size = step_size
        self.noise_std = noise_std
        self.eot_samples = eot_samples  # Number of forward passes for EOT

    def langevin_sample(self, x):
        if self.num_steps == 0:
            return x.detach()
    
        x = x.detach().clone()
        x.requires_grad_(True)  # Enable gradients for Langevin sampling
        for _ in range(self.num_steps):
            logits = self.base(x)
            energy = -torch.logsumexp(logits, dim=1).sum()  # negative log p(x)
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]
            noise = torch.randn_like(x) * self.noise_std
            x = x - 0.5 * self.step_size ** 2 * grad + self.step_size * noise
            x = x.detach().requires_grad_(True)
        return x.detach()

    def forward(self, x):
        logits_accum = 0
        for _ in range(self.eot_samples):
            x_sampled = self.langevin_sample(x)
            logits = self.base(x_sampled)
            logits_accum += logits
        return logits_accum / self.eot_samples


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
jem_base = CNN(out_dim=10)
jem_base.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
jem_base.to(device)
jem_base.eval()

jem0_model = JEMWrapper(jem_base, num_steps=0, step_size=0.5, noise_std=0.01, eot_samples=1)
jem1_model = JEMWrapper(jem_base, num_steps=1, step_size=0.5, noise_std=0.01, eot_samples=1)
jem10_model = JEMWrapper(jem_base, num_steps=10, step_size=0.5, noise_std=0.01, eot_samples=1)

# Create foolbox model for baseline
fmodel_baseline = fb.PyTorchModel(baseline_model, bounds=(-1, 1))

# Create foolbox models for JEM variants
fmodel_jem0 = fb.PyTorchModel(jem0_model, bounds=(-1, 1))
fmodel_jem1 = fb.PyTorchModel(jem1_model, bounds=(-1, 1))
fmodel_jem10 = fb.PyTorchModel(jem10_model, bounds=(-1, 1))

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
    for images, labels in dataloader:
        images = torch.tensor(images, requires_grad=True).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Compute and print accuracies
baseline_accuracy = compute_accuracy(baseline_model, test_loader)
jem0_accuracy = compute_accuracy(jem0_model, test_loader)
jem1_accuracy = compute_accuracy(jem1_model, test_loader)
jem10_accuracy = compute_accuracy(jem10_model, test_loader)

print(f"Baseline model accuracy: {baseline_accuracy:.2f}%")
print(f"JEM0 model accuracy: {jem0_accuracy:.2f}%")
print(f"JEM1 model accuracy: {jem1_accuracy:.2f}%")
print(f"JEM10 model accuracy: {jem10_accuracy:.2f}%")

# Get a batch of test images and labels
test_iter = iter(test_loader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)


linf_attack = fb.attacks.LinfPGD()
linf_epsilons = [0.1, 0.2, 0.3, 0.4]  # Typical range for L-inf
l2_attack = fb.attacks.L2PGD()
l2_epsilons = [1.0, 2.0, 3.0, 4.0]  # Typical range for L2

def run_attacks(fmodel, model_name, images, labels, linf_epsilons = [0.1, 0.2, 0.3, 0.4], l2_epsilons = [1.0, 2.0, 3.0, 4.0], sampling_steps=0):
    if sampling_steps is None:
        x_sampled = images.clone().detach().requires_grad_().to(device)
    elif sampling_steps > 0:
        x_sampled = jem_sample(fmodel, images, n_steps=sampling_steps)
    elif sampling_steps == 0:
        x_sampled = images
    
    
    # L-infinity PGD attack
    print(f"\nL-infinity PGD Attack Results on {model_name} Model:")
    for eps in linf_epsilons:
        _, adversarial, success = linf_attack(fmodel, x_sampled, labels, epsilons=eps)
        
        adv_outputs = fmodel(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
        record_results(model_name, "Linf", eps, adv_accuracy)
        print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

    # L2 PGD attack
    print(f"\nL2 PGD Attack Results on {model_name} Model:")
    for eps in l2_epsilons:
        _, adversarial, success = l2_attack(fmodel, x_sampled, labels, epsilons=eps)
        
        adv_outputs = fmodel(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
        record_results(model_name, "L2", eps, adv_accuracy)
        print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

run_attacks(fmodel_baseline, "baseline", images, labels, sampling_steps=None)
run_attacks(fmodel_jem0, "jem-0", images, labels)
run_attacks(fmodel_jem1, "jem-1", images, labels, sampling_steps=1)
run_attacks(fmodel_jem10, "jem-10", images, labels, sampling_steps=10)

df = pd.DataFrame(results)

# Save results to CSV
df.to_csv('adversarial_robustness_results.csv', index=False)
print("Results saved to 'adversarial_robustness_results.csv'")