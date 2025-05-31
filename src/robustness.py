import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd

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
jem0_model = CNN(out_dim=10)
jem0_model.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
jem0_model.to(device)
jem0_model.eval()

# Define function to do 1 step of Langevin dynamics
def sample_input_langevin(x, model, step_size=0.1, noise_scale=0.01):
    x = x.clone().detach().requires_grad_(True)
    energy = model(x).logsumexp(dim=1).sum()  # negative log-likelihood energy
    grad = torch.autograd.grad(energy, x)[0]
    x = x + step_size * grad + noise_scale * torch.randn_like(x)
    return x.detach()

# Load the JEM-1 model (with 1 Langevin sampling step)
jem1_model = CNN(out_dim=10)
jem1_model.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))  # same weights as jem0
jem1_model.to(device)
jem1_model.eval()

# Langevin dynamics sampling function
def jem_sample(model, x, n_steps=10, step_size=0.1, noise_std=0.005):
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

# Wrapper model that includes sampling before forward
class JEMWrapper(nn.Module):
    def __init__(self, model, n_steps):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, x):
        x_sampled = jem_sample(self.model, x, n_steps=self.n_steps)
        return self.model(x_sampled)

# Load the same model weights as JEM-0
jem10_base = CNN(out_dim=10)
jem10_base.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
jem10_base.to(device)
jem10_base.eval()

# Wrap in sampling module
jem10_model = JEMWrapper(jem10_base, n_steps=10)
jem10_model.eval()

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
jem_accuracy = compute_accuracy(jem0_model, test_loader)

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
    
    record_results("baseline", "Linf", eps, adv_accuracy_base_Linf)

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
    
    record_results("baseline", "L2", eps, adv_accuracy_base_L2)

    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy_base_L2:.2f}%")

# Create foolbox models for JEM variants
fmodel_jem0 = fb.PyTorchModel(jem0_model, bounds=(-1, 1))
fmodel_jem1 = fb.PyTorchModel(jem1_model, bounds=(-1, 1))

# PGD Attack on JEM-0
print("\nL-infinity PGD Attack Results on JEM-0 Model:")
for eps in linf_epsilons:
    _, adversarial, success = linf_attack(fmodel_jem0, images, labels, epsilons=eps)
    with torch.no_grad():
        adv_outputs = jem0_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
    
    record_results("jem-0", "Linf", eps, adv_accuracy)

    print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

print("\nL2 PGD Attack Results on JEM-0 Model:")
for eps in l2_epsilons:
    _, adversarial, success = l2_attack(fmodel_jem0, images, labels, epsilons=eps)
    with torch.no_grad():
        adv_outputs = jem0_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
    
    record_results("jem-0", "L2", eps, adv_accuracy)

    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

# PGD Attack on JEM-1 (use Langevin sampled input)
print("\nL-infinity PGD Attack Results on JEM-1 Model:")
for eps in linf_epsilons:
    _, adversarial, success = linf_attack(fmodel_jem1, images, labels, epsilons=eps)
    adversarial = sample_input_langevin(adversarial, jem1_model)
    with torch.no_grad():
        adv_outputs = jem1_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
    
    record_results("jem-1", "Linf", eps, adv_accuracy)

    print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

print("\nL2 PGD Attack Results on JEM-1 Model:")
for eps in l2_epsilons:
    _, adversarial, success = l2_attack(fmodel_jem1, images, labels, epsilons=eps)
    adversarial = sample_input_langevin(adversarial, jem1_model)
    with torch.no_grad():
        adv_outputs = jem1_model(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
    
    record_results("jem-1", "L2", eps, adv_accuracy)

    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

# Create Foolbox model for PGD attack
fmodel_jem10 = fb.PyTorchModel(jem10_base, bounds=(-1, 1))  # use base model, not wrapped


x_sampled = jem_sample(jem10_base, images, n_steps=10)  # sample before attack

# Run PGD attacks on sampled images
print("\nL-infinity PGD Attack Results on JEM-10:")
for eps in linf_epsilons:
    _, adversarial, success = linf_attack(fmodel_jem10, x_sampled, labels, epsilons=eps)
    
    with torch.no_grad():
        adv_outputs = jem10_base(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_jem10_Linf = (adv_predicted == labels).float().mean().item() * 100
    
    record_results("jem-10", "Linf", eps, adv_accuracy_jem10_Linf)
    
    print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy_jem10_Linf:.2f}%")


# Sample inputs using 10 steps of JEM sampling before the attack
x_sampled = jem_sample(jem10_base, images, n_steps=10)  # x_sampled is detached

# Use the base model (not the wrapped sampling model)
fmodel_jem10 = fb.PyTorchModel(jem10_base, bounds=(-1, 1))

# Run L2 PGD attack on pre-sampled inputs
print("\nL2 PGD Attack Results on JEM-10:")
for eps in l2_epsilons:  # list of L2 epsilon values (e.g., [0.1, 0.5, 1.0])
    _, adversarial, success = l2_attack(fmodel_jem10, x_sampled, labels, epsilons=eps)

    with torch.no_grad():
        adv_outputs = jem10_base(adversarial)
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_accuracy_jem10_L2 = (adv_predicted == labels).float().mean().item() * 100

    record_results("jem-10", "L2", eps, adv_accuracy_jem10_L2)
    
    print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy_jem10_L2:.2f}%")

df = pd.DataFrame(results)

# Save results to CSV
df.to_csv('adversarial_robustness_results.csv', index=False)
print("Results saved to 'adversarial_robustness_results.csv'")