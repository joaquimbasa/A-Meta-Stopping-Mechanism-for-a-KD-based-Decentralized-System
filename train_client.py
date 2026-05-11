import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
import json
import time
import sys
import os
import csv
import scipy.spatial.distance
from scipy.spatial import distance
from scipy.stats import entropy
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
import copy

# ============================================================================
# NEW IMPORT: Meta Early Stopping Module
# ============================================================================
from meta_early_stopping import MetaEarlyStopping, MetaStoppingConfig


# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Exclude the last FC layer
        self.fc1 = nn.Linear(base_model.fc.in_features, 10)  # First head
        self.fc2 = nn.Linear(base_model.fc.in_features, 10)  # Second head

    def forward(self, x):
        # Forward pass through the base model
        x = self.base(x)
        x = torch.flatten(x, 1)  # Flatten the output to feed into the FC layers

        # Compute outputs for both heads
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2


# All the existing divergence functions remain unchanged
def jensen_shannon_divergence(p, q):
    sumPQ = torch.add(p, q)
    tensor_mean = torch.mul(sumPQ, 0.5)
    comp1 = nn.KLDivLoss(reduction='sum',log_target=False)(tensor_mean.log(),p)
    comp2 = nn.KLDivLoss(reduction='sum',log_target=False)(tensor_mean.log(),q)
    sum_comp = comp1 + comp2
    jsd = torch.mul(sum_comp, 0.5)
    return jsd


def triangular_distance(x, y):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    if y.dim() == 1:
      y = y.unsqueeze(0)

    combined = x + y
    combined = torch.where(combined == 0, torch.ones_like(combined), combined)
    product = 2 * x * y
    distance = 1 - torch.sum(product / (combined),dim=1)
    return torch.sum(distance)


def compute_entropies(tensor):
    tensor = torch.where(tensor == 0, torch.ones_like(tensor), tensor)
    tensor_log = torch.log(tensor)
    tensor_mul = tensor * tensor_log
    entropy = -torch.sum(tensor_mul, dim=1)
    return entropy


def compute_complexity(tensor):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    entropies = compute_entropies(tensor)
    complexity = torch.exp(entropies)
    return complexity


def compute_sed(tensor1, tensor2):
    complexity1 = compute_complexity(tensor1)
    complexity2 = compute_complexity(tensor2)
    tensor_mean = (tensor1 + tensor2) / 2
    complexity12 = compute_complexity(tensor_mean)
    geometric_mean = torch.sqrt(complexity1 * complexity2)

    sed = torch.sum((complexity12 / geometric_mean) - 1)
    return sed


def compute_msed(tensors):
    complexities = [compute_complexity(tensor) for tensor in tensors]
    mean_tensor = torch.mean(torch.stack(tensors), 0)
    complexity_mean = compute_complexity(mean_tensor)
    geometric_mean = torch.prod(torch.stack(complexities), 0) ** (1 / len(complexities))
    msed = (1 / (len(complexities) - 1)) * (torch.sum(complexity_mean / geometric_mean-1) )

    return msed


def msed(tensors):
    n = len(tensors)

    # Step 1: Compute the average tensor (mean over all tensors)
    avg_tensor = torch.mean(torch.stack(tensors), dim=0)

    # Step 2: Compute the entropy of the average tensor
    entropy_avg = compute_entropies(avg_tensor)

    # Step 3: Compute the entropies of individual tensors and their average
    individual_entropies = torch.stack([compute_entropies(tensor) for tensor in tensors])
    avg_individual_entropy = torch.mean(individual_entropies, dim=0)

    # Step 4: Apply the MSED formula
    numerator = entropy_avg - avg_individual_entropy
    msed_value = torch.exp(numerator) - 1

    # Step 5: Scale by 1 / (n - 1)
    msed_value = msed_value / (n - 1)

    return torch.sum(msed_value)

#Define a cross entropy loss function
def cross_entropy_loss(p, q):
    q = torch.where(q == 0, torch.ones_like(q), q)
    return torch.sum(p * torch.log(q))


# implement a function to compute KL divergence and use - instead of / to avoid nan values


#all new functions
def shannon_entropy(tensor, dim=1):
    '''
    Compute shannon entropy from scratch
    '''
    entropy = -torch.sum(tensor * torch.log(tensor), dim=dim)
    return entropy


def compute_kl_torch(logit_target, logit_input, reduction='sum'):
   # p=target
   # q=input
    target_prob=nn.functional.softmax(logit_target, dim=1)
    input=nn.functional.log_softmax(logit_input, dim=1)
    return nn.KLDivLoss(reduction=reduction)(input,target_prob)



def sed(tensor1, tensor2):
    tensor_mean = torch.mul(tensor1 + tensor2, 0.5)
    entropy_tens_mean = compute_entropies(tensor_mean)
    mean_entropies = torch.mul(compute_entropies(tensor1) + compute_entropies(tensor2), 0.5)
    sed = torch.sum(torch.exp(entropy_tens_mean - mean_entropies) - 1)
    return sed


# ============================================================================
# NEW FUNCTION: Compute Meta-Validation Loss (Eq. 1 from formalization)
# ============================================================================

# ============================================================================
# NEW FUNCTION: Compute Meta-Validation Loss (Eq. 1 from formalization)
# ============================================================================
def evaluate_on_neighbors_validation(
    model,
    remote_clients_ids,
    dataset,
    num_clients,
    device,
    batch_size=128,
    worker=6
):
    

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    criterion = nn.CrossEntropyLoss(reduction='mean')

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for neighbor_id in remote_clients_ids:

            try:
                neighbor_valset = torchvision.datasets.ImageFolder(
                    root=f'.data/niid_{dataset}_S100_C{num_clients}/val/client_{neighbor_id}',
                    transform=transform_val
                )

                neighbor_valloader = DataLoader(
                    neighbor_valset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=worker
                )

                for inputs, labels in neighbor_valloader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    _, outputs2 = model(inputs)

                    loss = criterion(outputs2, labels)

                    batch_size_current = labels.size(0)

                    # accumulate weighted loss
                    total_loss += loss.item() * batch_size_current
                    total_samples += batch_size_current

            except Exception as e:
                print(f"Warning: Could not evaluate neighbor {neighbor_id}: {e}")
                continue

    if total_samples == 0:
        return 0.0

    meta_val_loss = total_loss / total_samples

    return meta_val_loss


# ============================================================================
# MAIN TRAINING LOOP WITH META EARLY STOPPING
# ============================================================================

seeds= [42]
for seed in seeds:

    # Set seed for Python's random number generator
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the value of temperature from JSON
    alphas = [0.5]
    num_clients = 8
    destinationClient = 8
    max_iteration = 40

    currentClients = [0,1, 2, 3, 4, 5, 6, 7]
    istest = True
    # chemins = ["sum_t_max","sum_pat","sum_div","sum_conv","sum_div_pat","sum_conv_pat","sum_conv_div","sum_conv_div_pat"]
    chemins = ["sum_pat","sum_div"]
    chemins = ["sum_t_max"]
    
    topology = "full"
    # currentClients = [4]

    dataset = "fashion_MNIST"
    dosum_values = [True]
    for chemin in chemins:
        # ========================================================================
        # NEW: Initialize Meta Early Stopping for all clients
        # ========================================================================
        print("\n" + "=" * 70)
        print("INITIALIZING META EARLY STOPPING")
        print("=" * 70)

        # Create configuration for meta early stopping

        if chemin == "sum_t_max":
            stopping_config = MetaStoppingConfig(
                P_patience=41,           # Wait 5 iterations for improvement
                epsilon_tol=10000000,      # Tolerance threshold
                n_div=40,                # Require 3 consecutive divergence iterations
                epsilon_conv=10000000,      # 1% convergence threshold (normalized by best loss)
                n_conv= 40,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_pat":
            stopping_config = MetaStoppingConfig(
                P_patience=5,           # Wait 5 iterations for improvement
                epsilon_tol=0.02,      # Tolerance threshold
                n_div=40,                # Require 3 consecutive divergence iterations
                epsilon_conv=10000000,      # 1% convergence threshold (normalized by best loss)
                n_conv= 40,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_div":
            stopping_config = MetaStoppingConfig(
                P_patience=40,           # Wait 5 iterations for improvement
                epsilon_tol=10000000,      # Tolerance threshold
                n_div=4,                # Require 3 consecutive divergence iterations
                epsilon_conv=10000000,      # 1% convergence threshold (normalized by best loss)
                n_conv= 40,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_conv":
            stopping_config = MetaStoppingConfig(
                P_patience=41,           # Wait 5 iterations for improvement
                epsilon_tol=10000000,      # Tolerance threshold
                n_div=40,                # Require 3 consecutive divergence iterations
                epsilon_conv=0.02,      # 1% convergence threshold (normalized by best loss)
                n_conv= 4,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_div_pat":
            stopping_config = MetaStoppingConfig(
                P_patience=5,           # Wait 5 iterations for improvement
                epsilon_tol=0.02,      # Tolerance threshold
                n_div=4,                # Require 3 consecutive divergence iterations
                epsilon_conv=10000000,      # 1% convergence threshold (normalized by best loss)
                n_conv= 40,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_conv_pat":
            stopping_config = MetaStoppingConfig(
                P_patience=5,           # Wait 5 iterations for improvement
                epsilon_tol=0.02,      # Tolerance threshold
                n_div=40,                # Require 3 consecutive divergence iterations
                epsilon_conv=0.02,      # 1% convergence threshold (normalized by best loss)
                n_conv= 4,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )

        elif chemin == "sum_conv_div":
            stopping_config = MetaStoppingConfig(
                P_patience=40,           # Wait 5 iterations for improvement
                epsilon_tol=10000000,      # Tolerance threshold
                n_div=4,                # Require 3 consecutive divergence iterations
                epsilon_conv=0.02,      # 1% convergence threshold (normalized by best loss)
                n_conv= 4,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        elif chemin == "sum_conv_div_pat":
            stopping_config = MetaStoppingConfig(
                P_patience=5,           # Wait 5 iterations for improvement
                epsilon_tol=0.02,      # Tolerance threshold
                n_div=4,                # Require 3 consecutive divergence iterations
                epsilon_conv=0.02,      # 1% convergence threshold (normalized by best loss)
                n_conv= 4,               # Require 3 consecutive convergence iterations
                t_max=40,               # Maximum 10 iterations
                use_convergence=True,   # Enable convergence criterion C3
                save_history=True       # Save loss history for analysis
            )
        
        
        
        
            
        

        print(f"Configuration:")
        print(f"  P_patience: {stopping_config.P_patience}")
        print(f"  n_div (divergence): {stopping_config.n_div}")
        print(f"  epsilon_conv: {stopping_config.epsilon_conv}")
        print(f"  t_max: {stopping_config.t_max}")

        # Dictionary to track stopping state for each client
        client_stoppers = {}
        client_stopped = {}

        # Initialize stoppers for all clients
        for client_id in currentClients:
            client_stoppers[client_id] = MetaEarlyStopping(
                client_id=client_id,
                config=stopping_config
            )
            client_stopped[client_id] = False

        print(f"Initialized meta early stopping for {len(currentClients)} clients")
        print("=" * 70 + "\n")
        # ========================================================================
        
        # MODIFIED: Increase max iterations since we now have early stopping
        for iteration in range(1, max_iteration+1):  # Changed from range(1, 6) to range(1, 11)
            if istest == False:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration}")
                print(f"{'='*70}")

                for currentClient in currentClients:

                    # ================================================================
                    # NEW: Skip clients that have already stopped
                    # ================================================================
                    if client_stopped[currentClient]:
                        print(f"\nClient {currentClient}: ALREADY STOPPED at iteration {client_stoppers[currentClient].stop_iteration}")
                        print(f"  Reason: {client_stoppers[currentClient].stop_reason}")
                        continue
                    # ================================================================

                    print(f'\n******Client******: {currentClient}')
                    #remoteClients = [0, 1, 2, 3,4, 5, 6, 7]
                    # Set up remote clients list (same as original)
                    if currentClient == 0:
                        remoteClients = [1,2,5,4,7,3,6,currentClient]
                    elif currentClient == 1:
                        remoteClients = [0,4,7,3,6,2,5,currentClient]
                    elif currentClient == 2:
                        remoteClients = [0,5,3,6,4,1,7,currentClient]
                    elif currentClient == 3:
                        remoteClients = [7,6,2,1,5,0,4,currentClient]
                    elif currentClient == 4:
                        remoteClients = [1, 5, 6, 0,2,3,7,currentClient]
                    elif currentClient == 5:
                        remoteClients = [2,4,0,7,3,6,1,currentClient]
                    elif currentClient == 6:
                        remoteClients = [3,4,7,2,1,5,0,currentClient]
                    elif currentClient == 7:
                        remoteClients = [3,1,6,5,0,2,4,currentClient]

                    # Load configuration from alpha4.json
                    with open('alpha4.json') as f:
                        data = json.load(f)
                        temperatures = data['temperature']
                        modes = data['mode']

                    for dosum in dosum_values:
                        if dosum == True:
                            distillation_case = chemin
                        else:
                            distillation_case = 'avg'

                        for mode in modes:
                            print(f'******Mode******: {mode}')
                            print()
                            for temperature in temperatures:
                                print(f'******Temperature******: {temperature}')
                                for alpha in alphas:

                                    print(f'******Alpha******: {alpha}')

                                    # Define transformations
                                    transform_train = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

                                    transform_val = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

                                    # Load datasets
                                    trainset_head1 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/train/client_{currentClient}', transform=transform_train)
                                    valset_head1 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/val/client_{currentClient}', transform=transform_val)
                                    trainset_head2 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/train/client_{currentClient}', transform=transform_train)
                                    valset_head2 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/val/client_{currentClient}', transform=transform_val)

                                    # Define batch size and workers
                                    batch_size = 128
                                    worker = 6

                                    # Create data loaders
                                    trainloader_head1 = DataLoader(trainset_head1, batch_size=batch_size, shuffle=True, num_workers=worker)
                                    valloader_head1 = DataLoader(valset_head1, batch_size=batch_size, shuffle=False, num_workers=worker)
                                    trainloader_head2 = DataLoader(trainset_head2, batch_size=batch_size, shuffle=True, num_workers=worker)
                                    valloader_head2 = DataLoader(valset_head2, batch_size=batch_size, shuffle=False, num_workers=worker)

                                    # Define student model
                                    if currentClient == 0 or currentClient == 4 or currentClient == 7:
                                        base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                                        student_model = CustomResNet(base_model)
                                        student_model.to(device)
                                    if currentClient == 1 or currentClient == 5 or currentClient == 6:
                                        base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                                        student_model = CustomResNet(base_model)
                                        student_model.to(device)
                                    if currentClient == 2 or currentClient == 3:
                                        base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                                        student_model = CustomResNet(base_model)
                                        student_model.to(device)

                                    # Define loss function
                                    criterion = nn.CrossEntropyLoss()

                                    # Define optimizer and scheduler
                                    optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

                                    # Define training parameters
                                    num_epochs = 50
                                    patience = 5
                                    best_val_loss = float('inf')
                                    no_improvement_count = 0

                                    # ====================================================
                                    # FIRST HEAD TRAINING (UNCHANGED)
                                    # ====================================================
                                    if not os.path.exists(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth'):
                                        print("Training first head...")
                                        for epoch in range(num_epochs):
                                            student_model.train()
                                            running_train_loss = 0.0
                                            correct_train = 0
                                            total_train = 0
                                            with tqdm(trainloader_head1, unit="batch") as tepoch:
                                                for inputs, labels in tepoch:
                                                    inputs, labels = inputs.to(device), labels.to(device)
                                                    optimizer.zero_grad()
                                                    outputs1, _ = student_model(inputs)
                                                    loss1 = criterion(outputs1, labels)
                                                    loss1.backward()
                                                    optimizer.step()
                                                    running_train_loss += loss1.item()
                                                    _, predicted1 = torch.max(outputs1, 1)
                                                    total_train += labels.size(0)
                                                    correct_train += (predicted1 == labels).sum().item()
                                                    tepoch.set_postfix(loss=running_train_loss / total_train, accuracy=100. * correct_train / total_train)
                                            scheduler.step()
                                            student_model.eval()
                                            running_val_loss = 0.0
                                            correct_val = 0
                                            total_val = 0
                                            with torch.no_grad():
                                                for inputs, labels in valloader_head1:
                                                    inputs, labels = inputs.to(device), labels.to(device)
                                                    outputs1, _ = student_model(inputs)
                                                    loss1 = criterion(outputs1, labels)
                                                    running_val_loss += loss1.item()
                                                    _, predicted1 = torch.max(outputs1, 1)
                                                    total_val += labels.size(0)
                                                    correct_val += (predicted1 == labels).sum().item()
                                            avg_val_loss = running_val_loss / len(valloader_head1)
                                            val_accuracy = 100 * correct_val / total_val
                                            print(f"Epoch [{epoch + 1}/{num_epochs}] - Head 1, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                                            if avg_val_loss < best_val_loss:
                                                best_val_loss = avg_val_loss
                                                no_improvement_count = 0
                                                torch.save({'model_state_dict': student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                                print('Saving the best model for Head 1')
                                            else:
                                                no_improvement_count += 1
                                                if no_improvement_count >= patience:
                                                    print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                                    break

                                    # ====================================================
                                    # SECOND HEAD TRAINING (WITH META EARLY STOPPING)
                                    # ====================================================
                                    else:
                                        print("Training second head with meta early stopping...")
                                        
                                        # Helper functions (unchanged)
                                        def initialize_fc2_from_fc1(model):
                                            with torch.no_grad():
                                                fc1_weights = model.fc1.weight.data
                                                sorted_weights, _ = torch.sort(fc1_weights, dim=1)
                                                model.fc2.weight.data.copy_(sorted_weights)
                                                model.fc2.bias.data.zero_()

                                        def load_second_head_checkpoint(student_model):
                                            second_head_checkpoint_path = f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/second_head_client{currentClient}_alpha{alpha}_temp{temperature}_{mode}_iter{iteration-1}.pth'

                                            if os.path.exists(second_head_checkpoint_path) and iteration >  1:
                                                print("Loading checkpoint for the second head...")
                                                checkpoint = torch.load(second_head_checkpoint_path)
                                                student_model.load_state_dict(checkpoint['model_state_dict'])
                                            else:
                                                print("No existing second head checkpoint found, initializing second head.")
                                                checkpoint = torch.load(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                                student_model.load_state_dict(checkpoint['model_state_dict'])
                                                initialize_fc2_from_fc1(student_model)
                                            return student_model


                                        def load_student_model_clone(
                                            client_number,
                                            dataset,
                                            num_clients,
                                            device,
                                            iteration,
                                            client_stoppers,
                                            client_stopped
                                        ):
                                            # Base model selection (unchanged)
                                            base_model_clone = models.resnet18(weights=None)
                                            model_clone = CustomResNet(base_model_clone)
                                            model_clone.to(device)

                                            # Determine which iteration to load
                                            effective_iter = get_effective_iteration(
                                                client_id=client_number,
                                                current_iteration=iteration,
                                                client_stoppers=client_stoppers,
                                                client_stopped=client_stopped
                                            )

                                            # Load correct checkpoint
                                            if effective_iter == 0:
                                                checkpoint_path = (
                                                    f'./{dataset}/niid_S100_C{num_clients}/first_head_client{client_number}.pth'
                                                )
                                            else:
                                                checkpoint_path = (
                                                    f'./{dataset}/niid_S100_C{destinationClient}/'
                                                    f'clients_{distillation_case}_{topology}/'
                                                    f'second_head_client{client_number}_alpha{alpha}_temp{temperature}_{mode}_iter{effective_iter}.pth'
                                                )

                                            if not os.path.exists(checkpoint_path):
                                                raise FileNotFoundError(
                                                    f"Checkpoint not found for client {client_number} at iteration {effective_iter}")
                                                

                                            checkpoint = torch.load(checkpoint_path)
                                            model_clone.load_state_dict(checkpoint['model_state_dict'])

                                            return model_clone


                                        def get_effective_iteration(client_id, current_iteration, client_stoppers, client_stopped):
                                            
                                            if client_stopped.get(client_id, False):
                                                return client_stoppers[client_id].get_best_iteration()
                                            else:
                                                return current_iteration - 1


                                        # Load the model
                                        checkpoint = torch.load(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                        student_model.load_state_dict(checkpoint['model_state_dict'])

                                        # Initialize or load second head
                                        student_model = load_second_head_checkpoint(student_model)

                                        # Load remote clients
                                        

                                        remote_clients = [
                                                            load_student_model_clone(
                                                                client_number=client_number,
                                                                dataset=dataset,
                                                                num_clients=num_clients,
                                                                device=device,
                                                                iteration=iteration,
                                                                client_stoppers=client_stoppers,
                                                                client_stopped=client_stopped
                                                            )
                                                            for client_number in remoteClients
                                                        ]



                                        # Define optimizer for second head
                                        optimizer = optim.SGD(student_model.fc2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

                                        # Reset best_val_loss for this iteration
                                        best_val_loss = float('inf')
                                        no_improvement_count = 0

                                        # ============================================
                                        # TRAINING LOOP FOR SECOND HEAD
                                        # ============================================
                                        start_time = time.time()

                                        for epoch in range(num_epochs):
                                            # Training phase (UNCHANGED)
                                            student_model.train()
                                            running_train_loss = 0.0
                                            correct_train = 0
                                            total_train = 0

                                            with tqdm(trainloader_head2, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
                                                for inputs, labels in tepoch:
                                                    inputs, labels = inputs.to(device), labels.to(device)
                                                    optimizer.zero_grad()

                                                    # Compute forward pass for remote clients
                                                    remote_outputs = []
                                                    with torch.no_grad():
                                                        for remote_client in remote_clients:
                                                            if iteration == 1:
                                                                remote_output, _ = remote_client(inputs)
                                                                remote_outputs.append(remote_output.to(device))
                                                            else:
                                                                _, remote_output = remote_client(inputs)
                                                                remote_outputs.append(remote_output.to(device))

                                                    # Aggregate remote outputs
                                                    if dosum == False:
                                                        remote_output = torch.stack(remote_outputs).mean(dim=0)

                                                    _, outputs2 = student_model(inputs)
                                                    loss2 = criterion(outputs2, labels)

                                                    # Compute distillation loss (UNCHANGED)
                                                    if mode == 'CE':
                                                        if dosum:
                                                            distillation_loss = sum(
                                                                nn.CrossEntropyLoss(reduction='mean')(outputs2 / temperature, nn.functional.softmax(ro / temperature, dim=1))
                                                                for ro in remote_outputs
                                                            )
                                                        else:
                                                            distillation_loss = nn.CrossEntropyLoss(reduction='sum')(outputs2 / temperature, nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'log_ce':
                                                            if dosum:
                                                                distillation_loss = sum(
                                                                    nn.CrossEntropyLoss(reduction='sum')(nn.functional.log_softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                                    for ro in remote_outputs
                                                                )
                                                            else:
                                                                distillation_loss = nn.CrossEntropyLoss(reduction='sum')(nn.functional.log_softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'KL':
                                                        if dosum:
                                                            distillation_loss = sum(
                                                                compute_kl_torch(ro / temperature, outputs2 / temperature, reduction='sum')
                                                                for ro in remote_outputs
                                                            )
                                                        else:
                                                            distillation_loss = compute_kl_torch(remote_output / temperature, outputs2 / temperature, reduction='sum')
                                                    elif mode == 'JS':
                                                        if dosum:
                                                            distillation_loss = sum(
                                                                jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                                for ro in remote_outputs
                                                            )
                                                        else:
                                                            distillation_loss = jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'TD':
                                                        if dosum:
                                                            distillation_loss = sum(
                                                                triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                                for ro in remote_outputs
                                                            )
                                                        else:
                                                            distillation_loss = triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'SED_exp' :
                                                            if dosum:
                                                                distillation_loss = sum(
                                                                    sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                                    for ro in remote_outputs
                                                                )
                                                            else:
                                                                distillation_loss = sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'SED':
                                                        if dosum:
                                                            distillation_loss = sum(
                                                                compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                                for ro in remote_outputs
                                                            )
                                                        else:
                                                            distillation_loss = compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                                    elif mode == 'MSED_exp' :
                                                            if dosum:
                                                                msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                                distillation_loss = msed(msed_tensors)
                                                            else:
                                                                msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                                distillation_loss = msed(msed_tensors)
                                                    elif mode == 'MSED':
                                                        msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                        distillation_loss = compute_msed(msed_tensors)
                                                    else:
                                                        raise ValueError(f"Unknown mode: {mode}")

                                                    # Total loss
                                                    loss2 = alpha * loss2 + (1 - alpha) * distillation_loss
                                                    loss2.backward()
                                                    optimizer.step()
                                                    running_train_loss += loss2.item()

                                                    # Accuracy computation
                                                    _, predicted2 = torch.max(outputs2, 1)
                                                    total_train += labels.size(0)
                                                    correct_train += (predicted2 == labels).sum().item()

                                                    # Update progress bar
                                                    tepoch.set_postfix(loss=running_train_loss / total_train, accuracy=100. * correct_train / total_train)

                                            # ========================================
                                            # VALIDATION PHASE (MODIFIED - Added meta-loss computation)
                                            # ========================================
                                            student_model.eval()
                                            running_val_loss = 0.0
                                            correct_val = 0
                                            total_val = 0

                                            # Compute local validation loss
                                            with torch.no_grad():
                                                for inputs, labels in valloader_head2:
                                                    inputs, labels = inputs.to(device), labels.to(device)
                                                    _, outputs2 = student_model(inputs)
                                                    loss2 = criterion(outputs2, labels)
                                                    running_val_loss += loss2.item()
                                                    _, predicted2 = torch.max(outputs2, 1)
                                                    total_val += labels.size(0)
                                                    correct_val += (predicted2 == labels).sum().item()

                                            avg_val_loss = running_val_loss / len(valloader_head2)
                                            val_accuracy = 100 * correct_val / total_val

                                            # Print validation metrics
                                            print(f"Epoch [{epoch + 1}/{num_epochs}] - Head 2, "
                                                f"Val Loss: {avg_val_loss:.4f}, "
                                                f"Val Accuracy: {val_accuracy:.2f}%")

                                            # Save best model (UNCHANGED)
                                            if avg_val_loss < best_val_loss:
                                                best_val_loss = avg_val_loss
                                                no_improvement_count = 0

                                                # Save model checkpoint
                                                torch.save(
                                                    {'model_state_dict': student_model.state_dict(),
                                                    'optimizer_state_dict': optimizer.state_dict()},
                                                    f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/second_head_client{currentClient}_alpha{alpha}_temp{temperature}_{mode}_iter{iteration}.pth'
                                                )
                                                print('Saving the best model for Head 2')
                                            else:
                                                no_improvement_count += 1
                                                if no_improvement_count >= patience:
                                                    print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                                    break

                                        end_time = time.time()
                                        training_time = end_time - start_time

                                        # ============================================
                                        # NEW: META EARLY STOPPING CHECK
                                        # After epoch loop completes for this iteration
                                        # ============================================

                                        # Compute meta-validation loss (Eq. 1)
                                        print("\nComputing meta-validation loss...")
                                        meta_val_loss = evaluate_on_neighbors_validation(
                                            model=student_model,
                                            remote_clients_ids=remoteClients[:-1],  # Exclude self
                                            dataset=dataset,
                                            num_clients=num_clients,
                                            device=device,
                                            batch_size=batch_size,
                                            worker=worker
                                        )

                                        print(f"Meta-Validation Loss: {meta_val_loss:.4f}")
                                        print(f"Local Validation Loss: {best_val_loss:.4f}")

                                        # Update meta early stopping
                                        should_stop, stop_reason = client_stoppers[currentClient].update(
                                            L_val=best_val_loss,     # Best local validation loss
                                            L_meta=meta_val_loss     # Meta-validation loss
                                        )

                                        # Check if stopping criterion triggered
                                        if should_stop or iteration==max_iteration:
                                            client_stopped[currentClient] = True

                                            print("\n" + "=" * 70)
                                            print(f"META EARLY STOPPING TRIGGERED FOR CLIENT {currentClient}")
                                            print("=" * 70)
                                            print(f"Iteration: {iteration}")
                                            print(f"Reason: {stop_reason}")
                                            print(f"Best iteration: {client_stoppers[currentClient].get_best_iteration()}")
                                            print(f"Best local val loss: {client_stoppers[currentClient].L_val_best:.4f}")
                                            print(f"Best meta val loss: {client_stoppers[currentClient].L_meta_best:.4f}")
                                            print("=" * 70 + "\n")

                                            # Save stopping history
                                            history_path = f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/stopping_history_client{currentClient}_{mode}_alpha{alpha}_temp{temperature}_seed{seed}.json'
                                            os.makedirs(os.path.dirname(history_path), exist_ok=True)
                                            client_stoppers[currentClient].save_history(history_path)
                                            print(f"Stopping history saved to {history_path}\n")

                                        print(f'Training complete for the second head! (Time: {training_time:.2f}s)')
    

    
    
            
            
