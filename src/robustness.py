import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps_refine', type=int, default=0)
parser.add_argument('--n_classes',type=int,default=10)
parser.add_argument('--init_batch_size', type=int, default=128)
# attack
parser.add_argument('--attack_conf',  action='store_true')
parser.add_argument('--random_init',  action='store_true')
parser.add_argument('--threshold', type=float, default=.7)
parser.add_argument('--debug',  action='store_true')
parser.add_argument('--no_random_start',  action='store_true')
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--distance", type=str, default='both')
parser.add_argument("--n_steps_pgd_attack", type=int, default=40)
parser.add_argument("--start_batch", type=int, default=-1)
parser.add_argument("--end_batch", type=int, default=10)
parser.add_argument("--sgld_sigma", type=float, default=1e-2)
parser.add_argument("--n_dup_chains", type=int, default=5)
parser.add_argument("--sigma", type=float, default=.03)
parser.add_argument("--base_dir", type=str, default='./adv_results')

parser.add_argument('--exp_name', type=str, default='exp', help='saves everything in ?r/exp_name/')
args = parser.parse_args()
device = torch.device('cuda')
args_ = vars(args)
for key in args_.keys():
    print('{}:   {}'.format(key,args_[key]))

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

        self.last_dim = out_dim  # important for JEM to know output dim

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        features = x  # penultimate features
        logits = self.fc2(features)

        if return_features:
            return features, logits
        return logits

class EnergyModel(nn.Module):
    def __init__(self, out_dim=10):
        super(EnergyModel, self).__init__()
        self.f = CNN(out_dim)  # your custom CNN
        self.energy_output = nn.Linear(256, 1)  # use 256 = feature dim from CNN.fc1
        self.class_output = nn.Linear(256, out_dim)

    def forward(self, x, y=None):
        features, _ = self.f(x, return_features=True)
        return self.energy_output(features).squeeze()

    def classify(self, x):
        features, _ = self.f(x, return_features=True)
        return self.class_output(features)

class CCF(EnergyModel):
    def __init__(self):
        super(CCF, self).__init__(out_dim=10)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])

# Baseline CNN model (standard CNN without JEM)
class BaselineCNN(nn.Module):
    def __init__(self, out_dim=10):
        super(BaselineCNN, self).__init__()
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
        logits = self.fc2(x)
        return logits
    
    def classify(self, x):
        logits = self.forward(x)
        pred = logits.max(1)[1]
        return pred

class DummyModel(nn.Module):
    def __init__(self, f, n_steps_refine=0):
        super(DummyModel, self).__init__()
        self.f = f
        self.n_steps_refine = n_steps_refine

    def logits(self, x):
        return self.f.classify(x)

    def refined_logits(self, x, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps_refine
            
        if n_steps == 0:
            # No refinement, just return logits
            return self.logits(x)
            
        xs = x.size()
        dup_x = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, args.n_dup_chains, 1, 1, 1)
        dup_x = dup_x.view(xs[0] * args.n_dup_chains, xs[1], xs[2], xs[3])
        dup_x = dup_x + torch.randn_like(dup_x) * args.sigma
        refined = self.refine(dup_x, n_steps=n_steps, detach=False)
        logits = self.logits(refined)
        logits = logits.view(x.size(0), args.n_dup_chains, logits.size(1))
        logits = logits.mean(1)
        return logits

    def classify(self, x):
        logits = self.logits(x)
        pred = logits.max(1)[1]
        return pred

    def logpx_score(self, x):
        # unnormalized logprob, unconditional on class
        return self.f(x)

    def refine(self, x, n_steps=None, detach=True):
        if n_steps is None:
            n_steps = self.n_steps_refine
            
        # runs a markov chain seeded at x
        x_k = torch.autograd.Variable(x, requires_grad=True) if detach else x
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + args.sgld_sigma * torch.randn_like(x_k)
        final_samples = x_k.detach() if detach else x_k
        return final_samples

    def grad_norm(self, x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def logpx_delta_score(self, x, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps_refine
            
        # difference in logprobs from input x and samples from a markov chain seeded at x
        init_scores = self.f(x)
        x_r = self.refine(x, n_steps=n_steps)
        final_scores = self.f(x_r)
        return init_scores - final_scores

    def logp_grad_score(self, x):
        return -self.grad_norm(x)

# Custom PGD Attack Implementation
class CustomPGDAttack:
    def __init__(self, model, eps, alpha, steps, norm='Linf', random_start=True, is_baseline=False):
        """
        Custom PGD Attack
        
        Args:
            model: The model to attack
            eps: Maximum perturbation magnitude
            alpha: Step size
            steps: Number of PGD steps
            norm: 'Linf' or 'L2'
            random_start: Whether to start with random noise
            is_baseline: Whether the model is baseline CNN
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start
        self.is_baseline = is_baseline
        
    def project(self, x, x_orig):
        """Project perturbation to norm ball"""
        if self.norm == 'Linf':
            x = torch.max(torch.min(x, x_orig + self.eps), x_orig - self.eps)
        elif self.norm == 'L2':
            delta = x - x_orig
            delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
            factor = torch.min(torch.ones_like(delta_norms), self.eps / (delta_norms + 1e-12))
            delta = delta * factor.view(-1, 1, 1, 1)
            x = x_orig + delta
        
        # Clip to valid input range [-1, 1]
        x = torch.clamp(x, -1, 1)
        return x
    
    def attack(self, x, y):
        """
        Perform PGD attack
        
        Args:
            x: Input images
            y: True labels
            
        Returns:
            x_adv: Adversarial examples
        """
        if self.eps == 0:
            return x
            
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.random_start and self.eps > 0:
            if self.norm == 'Linf':
                noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            elif self.norm == 'L2':
                noise = torch.randn_like(x_adv)
                noise_norms = torch.norm(noise.view(noise.shape[0], -1), p=2, dim=1)
                noise = noise / (noise_norms.view(-1, 1, 1, 1) + 1e-12) * self.eps * torch.rand(x.shape[0], device=x.device).view(-1, 1, 1, 1)
            
            x_adv = x_adv + noise
            x_adv = self.project(x_adv, x)
        
        # PGD iterations
        for i in range(self.steps):
            x_adv.requires_grad_(True)
            
            # Get logits
            if self.is_baseline:
                logits = self.model(x_adv)
            else:
                logits = self.model.refined_logits(x_adv)
            
            # Compute loss
            loss = F.cross_entropy(logits, y)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # Update adversarial examples
            if self.norm == 'Linf':
                x_adv = x_adv.detach() + self.alpha * grad.sign()
            elif self.norm == 'L2':
                grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-12
                grad = grad / grad_norms.view(-1, 1, 1, 1)
                x_adv = x_adv.detach() + self.alpha * grad
            
            # Project back to norm ball
            x_adv = self.project(x_adv, x)
        
        return x_adv.detach()

# Load the test dataset
test_dataset = MNIST(root='./data', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: x * 2 - 1)
                     ]))

# Create test dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to evaluate clean accuracy
def evaluate_clean_accuracy(model, test_loader, device, model_name, is_baseline=False):
    correct = 0
    total = 0
    model.eval()
    
    print(f"\nEvaluating clean accuracy for {model_name}...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.start_batch != -1 and batch_idx < args.start_batch:
                continue
            if args.end_batch != -1 and batch_idx >= args.end_batch:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            if is_baseline:
                pred = model.classify(data)
            else:
                pred = model.classify(data)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"{model_name} clean accuracy: {accuracy:.2f}%")
    return accuracy

# Function to run adversarial attacks using custom PGD
def run_adversarial_attacks(model, test_loader, device, model_name, distance_type='Linf', is_baseline=False):
    print(f"\nRunning {distance_type} attacks on {model_name}...")
    
    # Define epsilon values and step sizes
    if distance_type == 'L2':
        epsilons = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
        # Step size as fraction of epsilon
        alpha_ratio = 0.1
    else:  # Linf
        epsilons = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0]
        # Step size as fraction of epsilon
        alpha_ratio = 0.1
    
    # Store results for each epsilon
    epsilon_results = {}
    
    for epsilon in epsilons:
        print(f"  Testing epsilon = {epsilon}")
        correct = 0
        total = 0
        
        # Calculate step size
        alpha = epsilon * alpha_ratio if epsilon > 0 else 0
        
        # Create attack instance
        attack = CustomPGDAttack(
            model=model,
            eps=epsilon,
            alpha=alpha,
            steps=args.n_steps_pgd_attack,
            norm=distance_type,
            random_start=not args.no_random_start,
            is_baseline=is_baseline
        )
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.start_batch != -1 and batch_idx < args.start_batch:
                continue
            if args.end_batch != -1 and batch_idx >= args.end_batch:
                break
                
            data, target = data.to(device), target.to(device)
            
            try:
                # Generate adversarial examples
                adv_data = attack.attack(data, target)
                
                # Evaluate on adversarial examples
                model.eval()
                with torch.no_grad():
                    if is_baseline:
                        pred = model.classify(adv_data)
                    else:
                        pred = model.classify(adv_data)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
            except Exception as e:
                print(f"    Error in attack for epsilon {epsilon}: {e}")
                continue
            
            if args.debug and batch_idx % 10 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f"    Batch {batch_idx}, Current accuracy: {current_acc:.2f}%")
        
        if total > 0:
            accuracy = 100. * correct / total
            epsilon_results[epsilon] = accuracy
            print(f"  Epsilon {epsilon}: Accuracy = {accuracy:.2f}%")
            
            # Record results
            record_results(model_name, f"{distance_type}-PGD", epsilon, accuracy)
        else:
            print(f"  No samples processed for epsilon {epsilon}")
    
    return epsilon_results

def load_and_create_models():
    """Load the trained models and create different configurations"""
    models = {}
    
    # 1. Baseline CNN (separate trained model)
    print("Loading baseline CNN model...")
    try:
        baseline_model = BaselineCNN(out_dim=10)
        baseline_model.load_state_dict(torch.load('baseline_model.pth', map_location=device))
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        models['Baseline CNN'] = (baseline_model, True)  # True indicates it's baseline
        print("✓ Baseline CNN loaded successfully")
    except FileNotFoundError:
        print("✗ baseline_model.pth not found. Skipping baseline CNN.")
    except Exception as e:
        print(f"✗ Error loading baseline CNN: {e}")
    
    # 2-4. JEM models with different refinement steps
    print("Loading trained JEM model...")
    try:
        ccf_model = CCF()
        ccf_model.load_state_dict(torch.load('mnist-run-4.pth', map_location=device))
        ccf_model = ccf_model.to(device)
        ccf_model.eval()
        
        # JEM with 0 refinement steps
        jem_0 = DummyModel(ccf_model, n_steps_refine=0)
        models['JEM (0 steps)'] = (jem_0, False)
        
        # JEM with 1 refinement step
        jem_1 = DummyModel(ccf_model, n_steps_refine=1)
        models['JEM (1 step)'] = (jem_1, False)
        
        # JEM with 10 refinement steps
        jem_10 = DummyModel(ccf_model, n_steps_refine=10)
        models['JEM (10 steps)'] = (jem_10, False)
        
        # JEM with 25 refinement steps
        jem_25 = DummyModel(ccf_model, n_steps_refine=25)
        models['JEM (25 steps)'] = (jem_25, False)
        
        print("✓ JEM models loaded successfully")
        
    except FileNotFoundError:
        print("✗ mnist-run-4.pth not found. Skipping JEM models.")
    except Exception as e:
        print(f"✗ Error loading JEM models: {e}")
    
    if not models:
        raise ValueError("No models could be loaded. Please check your model files.")
        
    return models

# Main evaluation
print("="*50)
print("ADVERSARIAL ROBUSTNESS EVALUATION")
print("="*50)

# Load and create all model configurations
models = load_and_create_models()

# Test each model configuration
for model_name, (model, is_baseline) in models.items():
    print(f"\n{'='*20} {model_name} {'='*20}")
    
    # Evaluate clean accuracy
    clean_acc = evaluate_clean_accuracy(model, test_loader, device, model_name, is_baseline)
    record_results(model_name, "Clean", 0.0, clean_acc)
    
    # Run L-infinity attacks
    if args.distance == 'Linf' or args.distance == 'both':
        linf_results = run_adversarial_attacks(model, test_loader, device, model_name, 'Linf', is_baseline)
    
    # Run L2 attacks  
    if args.distance == 'L2' or args.distance == 'both':
        l2_results = run_adversarial_attacks(model, test_loader, device, model_name, 'L2', is_baseline)

# Create and save results DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(results_df.to_string(index=False))

# Save results in multiple formats for easy plotting
os.makedirs(args.base_dir, exist_ok=True)

# 1. Save complete results CSV
results_path = os.path.join(args.base_dir, f"{args.exp_name}_comparative_adversarial_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nComplete results saved to: {results_path}")

# 2. Create separate CSV files for each norm (optimized for line plotting)
attack_types = results_df[results_df['Attack'] != 'Clean']['Attack'].unique()

for attack_type in attack_types:
    attack_data = results_df[results_df['Attack'] == attack_type]
    
    # Create pivot table for easy plotting
    pivot_table = attack_data.pivot(index='Epsilon', columns='Model', values='Adversarial Accuracy (%)')
    
    # Save pivot table
    norm_name = attack_type.replace('-PGD', '')
    pivot_path = os.path.join(args.base_dir, f"{args.exp_name}_{norm_name}_results_for_plotting.csv")
    pivot_table.to_csv(pivot_path)
    print(f"{norm_name} results for plotting saved to: {pivot_path}")

# 3. Create a summary file with clean accuracies
clean_data = results_df[results_df['Attack'] == 'Clean'][['Model', 'Adversarial Accuracy (%)']].copy()
clean_data.columns = ['Model', 'Clean Accuracy (%)']
clean_path = os.path.join(args.base_dir, f"{args.exp_name}_clean_accuracies.csv")
clean_data.to_csv(clean_path, index=False)
print(f"Clean accuracies saved to: {clean_path}")

# 4. Create a plotting-ready long format CSV
plot_data = []
for attack_type in attack_types:
    attack_data = results_df[results_df['Attack'] == attack_type]
    for _, row in attack_data.iterrows():
        plot_data.append({
            'Model': row['Model'],
            'Norm': attack_type.replace('-PGD', ''),
            'Epsilon': row['Epsilon'],
            'Accuracy': row['Adversarial Accuracy (%)']
        })

plot_df = pd.DataFrame(plot_data)
plot_path = os.path.join(args.base_dir, f"{args.exp_name}_plotting_data.csv")
plot_df.to_csv(plot_path, index=False)
print(f"Plotting-ready data saved to: {plot_path}")

print(f"\n📊 Files created for easy plotting:")
print(f"   • Line plots by norm: {args.exp_name}_[Linf/L2]_results_for_plotting.csv")
print(f"   • Combined plotting data: {args.exp_name}_plotting_data.csv")
print(f"   • Clean accuracies: {args.exp_name}_clean_accuracies.csv")

# Print comparative analysis
print("\n" + "="*50)
print("COMPARATIVE ANALYSIS")
print("="*50)

# Group results by attack type and epsilon for comparison
attack_types = results_df[results_df['Attack'] != 'Clean']['Attack'].unique()

for attack_type in attack_types:
    print(f"\n{attack_type} Attack Comparison:")
    attack_data = results_df[results_df['Attack'] == attack_type]
    
    # Create pivot table for easy comparison
    pivot = attack_data.pivot(index='Epsilon', columns='Model', values='Adversarial Accuracy (%)')
    print(pivot.to_string())
    
    # Find best performing model at highest epsilon
    max_epsilon = attack_data['Epsilon'].max()
    max_eps_data = attack_data[attack_data['Epsilon'] == max_epsilon]
    best_model = max_eps_data.loc[max_eps_data['Adversarial Accuracy (%)'].idxmax(), 'Model']
    best_acc = max_eps_data['Adversarial Accuracy (%)'].max()
    
    print(f"\nBest performing model at epsilon {max_epsilon}: {best_model} ({best_acc:.2f}%)")

# Print clean accuracy comparison
print(f"\nClean Accuracy Comparison:")
clean_data = results_df[results_df['Attack'] == 'Clean']
for _, row in clean_data.iterrows():
    print(f"  {row['Model']}: {row['Adversarial Accuracy (%)']:.2f}%")