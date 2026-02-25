# =============================================================================
# PROJET : GENÈSE DU DUBOSSON FEYNMAN ENGINE (DFE) - ÉDITION FINALE CONSOLIDÉE
# =============================================================================
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
import copy

# --- 1. ARCHITECTURE GÉNÉTIQUE UNIVERSELLE (CERVEAU NEURONAL) ---
class LifeGenome(nn.Module):
    def __init__(self, color, name):
        super().__init__()
        self.color = color
        self.name = name
        self.brain = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(), # La signature sigmoïde découverte
            nn.Linear(32, 1),
            nn.Tanh()
        )
    def forward(self, x): return self.brain(x)
    def mutate(self, rate=0.02):
        with torch.no_grad():
            for p in self.parameters(): p.add_(torch.randn(p.size(), device=p.device) * rate)

# --- 2. CONFIGURATION DE L'ÉCOSYSTÈME MULTI-ESPÈCES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POP = 50

# Initialisation des Populations
pop_A = [LifeGenome('#00FFFF', 'Hôte Infecté').to(device) for _ in range(N_POP)]
pop_B = [LifeGenome('#FF4444', 'Témoin Sain').to(device) for _ in range(N_POP)]
virus = [LifeGenome('#550000', 'Virus').to(device) for _ in range(10)]

# Archives de l'évolution
history_A, history_B, history_V = [], [], []

print(f"🚀 Initialisation de la Genèse sur {device}...")
print("Règle de Vie : 10.0724 - (0.012 * Φ) - (78.38 * m) + (0.11 * V)")

# --- 3. BOUCLE DE VIE ÉTERNELLE ---
try:
    for gen in range(100000):
        # Dynamique du Champ de Feynman (Environnement)
        phi = 0.2 * torch.sin(torch.tensor(gen / 60.0, device=device))
        
        # Fonction d'évaluation de la survie
        def evaluate(pop, p_val):
            return [(1.0 - torch.abs(p_val - agent(torch.tensor([p_val.item(), 0.1], device=device).float())).item()) for agent in pop]

        # Calcul des Fitness
        fit_A = evaluate(pop_A, phi)
        fit_B = evaluate(pop_B, phi)
        
        # Logique de Parasitisme (Le Virus décode l'Espèce A)
        fit_V = []
        for v in virus:
            v_action = v(torch.tensor([phi.item(), 0.1], device=device).float())
            target_A = pop_A[0](torch.tensor([phi.item(), 0.1], device=device).float()) # Cible le premier de A
            v_score = 1.0 - torch.abs(target_A - v_action)
            fit_V.append(v_score.item())
            # Effet de l'infection sur l'espèce A
            if v_score > 0.92: 
                fit_A = [f - 0.015 for f in fit_A]

        # Sélection Naturelle Élitiste (Transmission du savoir)
        def natural_selection(pop, scores, m_rate=0.02):
            best_dna = pop[scores.index(max(scores))].state_dict()
            for i, agent in enumerate(pop):
                if scores[i] < 0.88: # Seuil de survie strict
                    agent.load_state_dict(copy.deepcopy(best_dna))
                    agent.mutate(m_rate)

        natural_selection(pop_A, fit_A, 0.02)
        natural_selection(pop_B, fit_B, 0.02)
        natural_selection(virus, fit_V, 0.05) # Le virus évolue plus vite

        # Visualisation Temps Réel (Toutes les 200 générations)
        if gen % 200 == 0:
            history_A.append(sum(fit_A)/N_POP)
            history_B.append(sum(fit_B)/N_POP)
            history_V.append(sum(fit_V)/10)
            
            display.clear_output(wait=True)
            plt.figure(figsize=(12, 6))
            plt.plot(history_A, color='#00FFFF', label='Espèce A (Hôte Infecté)')
            plt.plot(history_B, color='#FF4444', label='Espèce B (Témoin Sain)')
            plt.plot(history_V, color='#550000', linestyle='--', alpha=0.7, label='Virus (Parasite Informationnel)')
            plt.title(f"Genèse DFE | Génération : {gen} | Équilibre de la Reine Rouge", fontsize=14)
            plt.ylabel("Indice de Vie (Fitness)")
            plt.ylim(0.4, 1.1)
            plt.legend(loc='lower left')
            plt.grid(alpha=0.1)
            plt.show()

except KeyboardInterrupt:
    print("\n⌛ Genèse mise en pause. Le code source est préservé.")
