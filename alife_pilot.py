import numpy as np
from scipy.integrate import solve_ivp
import pyscf
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalwCommandline
from Bio import AlignIO
import torch
import matplotlib.pyplot as plt
import os

# Optimized Parameters with Vectorization and GPU Support
N = 1024  # Grid size (32x32)
dt = 0.1  # Time step (ms)
generations = 5000000  # Max generations (extended)
cp_bias = 0.0245  # Real CP asymmetry central value (LHCb 2025)
scalar_energy = 0.0075  # Eternal scalar field energy density
mem_gain = 0.8  # Boosted with DNA/RNA
senescence_threshold = 0.8  # For programmed death
protein_boost = 0.2  # Protein modulation for neural density
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available

# Initial State with New Traits (Vectorized)
V = np.full(N, -70.0)
mem = np.zeros(N)
resistance = np.full(N, 0.08)
innovation = np.full(N, 0.0)
cooperation = np.full(N, 0.0)
neural_density = np.full(N, 0.0)
adaptability = np.full(N, 0.0)
quantum_coherence = np.full(N, 0.0)
senescence = np.full(N, 0.0)  # New senescence trait
species = np.random.choice([0, 1, 2], size=N, p=[0.32, 0.35, 0.33])

# Move to Torch Tensors for GPU Acceleration
V_t = torch.tensor(V, dtype=torch.float32, device=device)
mem_t = torch.tensor(mem, dtype=torch.float32, device=device)
resistance_t = torch.tensor(resistance, dtype=torch.float32, device=device)
innovation_t = torch.tensor(innovation, dtype=torch.float32, device=device)
cooperation_t = torch.tensor(cooperation, dtype=torch.float32, device=device)
neural_density_t = torch.tensor(neural_density, dtype=torch.float32, device=device)
adaptability_t = torch.tensor(adaptability, dtype=torch.float32, device=device)
quantum_coherence_t = torch.tensor(quantum_coherence, dtype=torch.float32, device=device)
senescence_t = torch.tensor(senescence, dtype=torch.float32, device=device)
species_t = torch.tensor(species, dtype=torch.int32, device=device)

# Eternal Scalar Field (Vectorized)
def scalar_field(t, size=N):
    return scalar_energy * np.sin(t) + np.random.normal(0, 0.1, size)

# Vectorized Protocell Dynamics (Using Torch for Batch Processing)
def vectorized_dynamics(V, m, mem, I_ext, g_Ca_base, mem_gain, coherence, protein_boost, sen):
    m_inf = 1 / (1 + torch.exp((-V - 30) / 9.5))
    tau_m = 1 / (0.0125 * torch.exp((-V - 65) / 20) + 0.0125 * torch.exp((V + 35) / 80))
    g_Ca = g_Ca_base * m**3 * (1 + mem_gain * mem) * (1 + coherence) * (1 + protein_boost) * (1 - sen)
    I_Ca = g_Ca * (V - 120)
    I_L = 0.1 * (V + 70)
    dV = (-I_L - I_Ca + I_ext) / 1.0
    dm = (m_inf - m) / tau_m
    dmem = 0.01 * m - 0.001 * mem
    return dV, dm, dmem

# Simple Neural Net (PyTorch) for Cognition (Optimized with Batch)
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

net = NeuralNet().to(device)

# Protein Reaction Example (RDKit, Cached for Optimization)
def generate_protein_boost():
    amino_acid = Chem.MolFromSmiles('NCC(=O)O')  # Glycine
    chain = AllChem.ReactionFromSmarts('[N:1].[C:2](=O)[O:3]>>[N:1][C:2](=O)[O:3]')
    products = chain.RunReactants((amino_acid, amino_acid))
    if products:
        return len(Chem.MolToSmiles(products[0][0])) / 10
    return 0.2

protein_boost = generate_protein_boost()

# Evolution Loop with Vectorization and GPU
fitness_history = []
population_history = [N]

for gen in range(1, generations + 1):
    # Perturbations (Vectorized)
    if gen % 5 == 0:
        severity = np.random.uniform(0.5, 0.8)
        affected = np.random.choice(N, int(severity * N), replace=False)
        V_t[affected] -= torch.tensor(np.random.uniform(20, 50, len(affected)), device=device)

    # Dynamics (Batch Process on GPU)
    I_ext = torch.tensor(np.random.uniform(0, 20, N), device=device)
    m = torch.full((N,), 0.5, device=device)
    dV, dm, dmem = vectorized_dynamics(V_t, m, mem_t, I_ext, 1.0, mem_gain, quantum_coherence_t, protein_boost, senescence_t)
    V_t += dV * dt
    m += dm * dt
    mem_t += dmem * dt

    # Fitness with CP Bias (Vectorized)
    fitness = torch.clamp(V_t / -70, 0, 1) + cp_bias

    # Senescence Evolution (Vectorized)
    senescence_t += 0.001
    senescence_t = torch.clamp(senescence_t, 0, 1)
    senescence_death = senescence_t > senescence_threshold
    N -= senescence_death.sum().item()  # Death frees resources

    # Mutations (Vectorized with Torch)
    mutate_idx = torch.rand(N, device=device) < 0.12
    resistance_t[mutate_idx] += torch.normal(mean=0.0, std=0.01, size=mutate_idx.sum(), device=device)
    # Similar for other traits...

    # Sharing via Cooperation (Optimized Neighbor Sampling)
    for i in np.random.choice(N, int(N * 0.2)):  # Sample subset for speed
        if cooperation_t[i] > 0.2:
            neighbors = torch.randint(0, N, (4,), device=device)
            resistance_t[i] = resistance_t[torch.cat((torch.tensor([i], device=device), neighbors))].mean()

    # Neural Decision (Batch on GPU)
    inputs = fitness.unsqueeze(1)
    decisions = net(inputs).squeeze().detach().cpu().numpy()

    # Adaptability and Quantum (Vectorized)
    adaptability_t += torch.tensor(decisions, device=device) * 0.01
    quantum_coherence_t += torch.normal(mean=0.0, std=0.01, size=(N,), device=device) * adaptability_t.mean()

    # Replication/Death (Vectorized)
    replicate = fitness > 0.35
    death = fitness < 0.3
    N += replicate.sum().item() - death.sum().item()
    N = max(1000, min(N, 30000))

    # Scalar Shock (Vectorized)
    shock = scalar_field(gen, N)
    V_t += torch.tensor(shock, device=device) * torch.rand(N, device=device) * 0.1 - 0.05  # Optimized random

    fitness_history.append(fitness.mean().item())
    population_history.append(N)

    if gen % 100000 == 0:
        print(f"Generation {gen}: Population {N}, Avg Fitness {fitness.mean().item()}")

# Save Optimized Results
np.save('fitness_history.npy', fitness_history)
np.save('population_history.npy', population_history)

# Visualization (Efficient Plotting)
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(generations), fitness_history)
ax[0].set_title('Fitness Evolution')
ax[1].plot(np.arange(generations + 1), population_history)
ax[1].set_title('Population Evolution')
plt.tight_layout()
plt.savefig('evolution_plot.png')
plt.show()
""")</parameter>
</xai:function_call>