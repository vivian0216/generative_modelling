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

dict = torch.load('mnist-run-4.pth', map_location=device)
print("Model loaded with keys:", dict.keys())

f = CCF()
print("Loading model weights...")
f.load_state_dict(torch.load('mnist-run-4.pth', map_location=device))

f = DummyModel(f)
model = f.to(device)
model.eval()
labels = torch.arange(0, 10).to(device)
criterion = fb.criteria.Misclassification(labels)

model_wrapped = gradient_attack_wrapper(model)
fmodel = fb.models.PyTorchModel(model_wrapped, bounds=(0., 1.), device=device)

if args.distance == 'L2':
    distance = fb.distances.LpDistance(p=2)
    attack = fb.attacks.L2BasicIterativeAttack()
else:
    # Linf
    distance = fb.distances.LpDistance(p=float('inf'))
    attack = fb.attacks.LinfProjectedGradientDescentAttack(random_start=True)

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

# Load the baseline CNN model
# baseline_model = CNN(out_dim=10)  # MNIST has 10 classes
# baseline_model.load_state_dict(torch.load('./models/baseline_model.pth', map_location=device))
# baseline_model.to(device)
# baseline_model.eval()

# # Load the JEM-0 model (no sampling)
# jem_base = CNN(out_dim=10)
# jem_base.load_state_dict(torch.load('mnist-run-3.pth', map_location=device))
# jem_base.to(device)
# jem_base.eval()

# jem0_model = JEMWrapper(jem_base, num_steps=0, step_size=0.5, noise_std=0.01, eot_samples=1)
# jem1_model = JEMWrapper(jem_base, num_steps=1, step_size=0.5, noise_std=0.01, eot_samples=1)
# jem10_model = JEMWrapper(jem_base, num_steps=10, step_size=0.5, noise_std=0.01, eot_samples=1)

# # Create foolbox model for baseline
# fmodel_baseline = fb.PyTorchModel(baseline_model, bounds=(-1, 1))

# # Create foolbox models for JEM variants
# fmodel_jem0 = fb.PyTorchModel(jem0_model, bounds=(-1, 1))
# fmodel_jem1 = fb.PyTorchModel(jem1_model, bounds=(-1, 1))
# fmodel_jem10 = fb.PyTorchModel(jem10_model, bounds=(-1, 1))

# print("Models loaded successfully!")
# print(f"Test dataset size: {len(test_dataset)}")



# Function to compute accuracy
# def compute_accuracy(model, dataloader):
#     correct = 0
#     total = 0
#     for images, labels in dataloader:
#         images = torch.tensor(images, requires_grad=True).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     return 100 * correct / total

# # Compute and print accuracies
# baseline_accuracy = compute_accuracy(baseline_model, test_loader)
# jem0_accuracy = compute_accuracy(jem0_model, test_loader)
# jem1_accuracy = compute_accuracy(jem1_model, test_loader)
# jem10_accuracy = compute_accuracy(jem10_model, test_loader)

# print(f"Baseline model accuracy: {baseline_accuracy:.2f}%")
# print(f"JEM0 model accuracy: {jem0_accuracy:.2f}%")
# print(f"JEM1 model accuracy: {jem1_accuracy:.2f}%")
# print(f"JEM10 model accuracy: {jem10_accuracy:.2f}%")

# # Get a batch of test images and labels
# test_iter = iter(test_loader)
# images, labels = next(test_iter)
# images, labels = images.to(device), labels.to(device)


# linf_attack = fb.attacks.LinfPGD()
# linf_epsilons = [0.1, 0.2, 0.3, 0.4]  # Typical range for L-inf
# l2_attack = fb.attacks.L2PGD()
# l2_epsilons = [1.0, 2.0, 3.0, 4.0]  # Typical range for L2

# def run_attacks(fmodel, model_name, images, labels, linf_epsilons = [0.1, 0.2, 0.3, 0.4], l2_epsilons = [1.0, 2.0, 3.0, 4.0], sampling_steps=0):

#     # L-infinity PGD attack
#     print(f"\nL-infinity PGD Attack Results on {model_name} Model:")
#     for eps in linf_epsilons:
#         _, adversarial, success = linf_attack(fmodel, images, labels, epsilons=eps)
        
#         adv_outputs = fmodel(adversarial)
#         _, adv_predicted = torch.max(adv_outputs, 1)
#         adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
#         record_results(model_name, "Linf", eps, adv_accuracy)
#         print(f"L-inf eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

#     # L2 PGD attack
#     print(f"\nL2 PGD Attack Results on {model_name} Model:")
#     for eps in l2_epsilons:
#         _, adversarial, success = l2_attack(fmodel, images, labels, epsilons=eps)
        
#         adv_outputs = fmodel(adversarial)
#         _, adv_predicted = torch.max(adv_outputs, 1)
#         adv_accuracy = (adv_predicted == labels).float().mean().item() * 100
        
#         record_results(model_name, "L2", eps, adv_accuracy)
#         print(f"L2 eps {eps}: Adversarial accuracy = {adv_accuracy:.2f}%")

# run_attacks(fmodel_baseline, "baseline", images, labels, sampling_steps=None)
# run_attacks(fmodel_jem0, "jem-0", images, labels)
# run_attacks(fmodel_jem1, "jem-1", images, labels, sampling_steps=1)
# run_attacks(fmodel_jem10, "jem-10", images, labels, sampling_steps=10)

# df = pd.DataFrame(results)

# # Save results to CSV
# df.to_csv('adversarial_robustness_results.csv', index=False)
# print("Results saved to 'adversarial_robustness_results.csv'")



# print('Starting...')
# for i, (img, label) in enumerate(test_loader):
#     adversaries = []
#     if i < args.start_batch:
#         continue
#     if i >= args.end_batch:
#       break
#     img = img.data.cpu().numpy()
#     logits = model_wrapped(torch.from_numpy(img[:, :, :, :]).to(device))
#     _, top = torch.topk(logits,k=2,dim=1)
#     top = top.data.cpu().numpy()
#     pred = top[:,0]
#     for k in range(len(label)):
#       im = img[k,:,:,:]
#       orig_label = label[k].data.cpu().numpy()
#       if pred[k] != orig_label:
#         continue
#       best_adv = None
#       for ii in range(20):
#           try:
#             adversarial = attack(im, label=orig_label, unpack=False, random_start=True, iterations=args.n_steps_pgd_attack) 
#             if ii == 0 or best_adv.distance > adversarial.distance:
#                 best_adv = adversarial
#           except:
#             continue
#       try:
#           adversaries.append((im, orig_label, adversarial.image, adversarial.adversarial_class))
#       except:
#           continue
#     adv_save_dir = os.path.join('adversarial_testing')
#     save_file = 'adversarials_batch_'+str(i)
#     if not os.path.exists(os.path.join(adv_save_dir,save_file)):
#         os.makedirs(os.path.join(adv_save_dir,save_file))
#     np.save(os.path.join(adv_save_dir,save_file),adversaries)
