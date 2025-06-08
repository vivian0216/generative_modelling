import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd
import foolbox as fb

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
parser.add_argument("--distance", type=str, default='Linf')
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


class gradient_attack_wrapper(nn.Module):
  def __init__(self, model):
    super(gradient_attack_wrapper, self).__init__()
    self.model = model.eval()

  def forward(self, x):
    x.requires_grad_()
    out = self.model.refined_logits(x)
    return out

  def eval(self):
    return self.model.eval()

class DummyModel(nn.Module):
    def __init__(self, f):
        super(DummyModel, self).__init__()
        self.f = f

    def logits(self, x):
        return self.f.classify(x)

    def refined_logits(self, x, n_steps=args.n_steps_refine):
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

    def refine(self, x, n_steps=args.n_steps_refine, detach=True):
        # runs a markov chain seeded at x, use n_steps=10
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

    def logpx_delta_score(self, x, n_steps=args.n_steps_refine):
        # difference in logprobs from input x and samples from a markov chain seeded at x
        #
        init_scores = self.f(x)
        x_r = self.refine(x, n_steps=n_steps)
        final_scores = self.f(x_r)
        # for real data final_score is only slightly higher than init_score
        return init_scores - final_scores

    def logp_grad_score(self, x):
        return -self.grad_norm(x)

# Load model
dict = torch.load('mnist-run-4.pth', map_location=device)
print("Model loaded with keys:", dict.keys())

f = CCF()
print("Loading model weights...")
f.load_state_dict(torch.load('mnist-run-4.pth', map_location=device))

f = DummyModel(f)
model = f.to(device)
model.eval()

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
def evaluate_clean_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.start_batch != -1 and batch_idx < args.start_batch:
                continue
            if args.end_batch != -1 and batch_idx >= args.end_batch:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            pred = model.classify(data)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"Clean accuracy: {accuracy:.2f}%")
    return accuracy

# Function to run adversarial attacks
def run_adversarial_attacks(model, test_loader, device, distance_type='Linf'):
    print(f"\nRunning {distance_type} attacks...")
    
    # Wrap model for Foolbox
    model_wrapped = gradient_attack_wrapper(model)
    fmodel = fb.models.PyTorchModel(model_wrapped, bounds=(-1., 1.), device=device)
    
    # Define epsilon values to test
    if distance_type == 'L2':
        epsilons = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        attack = fb.attacks.L2ProjectedGradientDescentAttack(
            steps=args.n_steps_pgd_attack,
            random_start=not args.no_random_start
        )
    else:  # Linf
        epsilons = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
        attack = fb.attacks.LinfProjectedGradientDescentAttack(
            steps=args.n_steps_pgd_attack,
            random_start=not args.no_random_start
        )
    
    # Store results for each epsilon
    epsilon_results = {}
    
    for epsilon in epsilons:
        print(f"\nTesting epsilon = {epsilon}")
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.start_batch != -1 and batch_idx < args.start_batch:
                continue
            if args.end_batch != -1 and batch_idx >= args.end_batch:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Skip if epsilon is 0 (clean accuracy)
            if epsilon == 0.0:
                with torch.no_grad():
                    pred = model.classify(data)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            else:
                # Generate adversarial examples
                try:
                    _, adversarials, success = attack(fmodel, data, target, epsilons=epsilon)
                    
                    # Evaluate on adversarial examples
                    with torch.no_grad():
                        pred = model.classify(adversarials)
                        correct += (pred == target).sum().item()
                        total += target.size(0)
                        
                except Exception as e:
                    print(f"Error in attack for epsilon {epsilon}: {e}")
                    continue
            
            if args.debug and batch_idx % 10 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f"Batch {batch_idx}, Current accuracy: {current_acc:.2f}%")
        
        if total > 0:
            accuracy = 100. * correct / total
            epsilon_results[epsilon] = accuracy
            print(f"Epsilon {epsilon}: Accuracy = {accuracy:.2f}%")
            
            # Record results
            record_results("JEM", f"{distance_type}-PGD", epsilon, accuracy)
        else:
            print(f"No samples processed for epsilon {epsilon}")
    
    return epsilon_results

# Main evaluation
print("="*50)
print("ADVERSARIAL ROBUSTNESS EVALUATION")
print("="*50)

# Evaluate clean accuracy
clean_acc = evaluate_clean_accuracy(model, test_loader, device)
record_results("JEM", "Clean", 0.0, clean_acc)

# Run L-infinity attacks
if args.distance == 'Linf' or args.distance == 'both':
    linf_results = run_adversarial_attacks(model, test_loader, device, 'Linf')

# Run L2 attacks
if args.distance == 'L2' or args.distance == 'both':
    l2_results = run_adversarial_attacks(model, test_loader, device, 'L2')

# Create and save results DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(results_df.to_string(index=False))

# Save results
os.makedirs(args.base_dir, exist_ok=True)
results_path = os.path.join(args.base_dir, f"{args.exp_name}_adversarial_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Print summary statistics
print("\n" + "="*30)
print("SUMMARY STATISTICS")
print("="*30)

for attack_type in results_df['Attack'].unique():
    if attack_type != 'Clean':
        attack_results = results_df[results_df['Attack'] == attack_type]
        print(f"\n{attack_type} Attack:")
        print(f"  Max epsilon tested: {attack_results['Epsilon'].max()}")
        print(f"  Accuracy at max epsilon: {attack_results['Adversarial Accuracy (%)'].min():.2f}%")
        print(f"  Accuracy drop from clean: {clean_acc - attack_results['Adversarial Accuracy (%)'].min():.2f}%")