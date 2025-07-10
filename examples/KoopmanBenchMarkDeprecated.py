#!/usr/bin/env python
# coding: utf-8
# ==============================================================
# BENCHMARK Cart-Pole · K-MPC (Koopman)  vs  N-MPC (fisico)
# ==============================================================
import os, time, torch, numpy as np, gym, matplotlib.pyplot as plt
from dataclasses import dataclass
from contextlib import redirect_stdout
from tqdm.auto import trange
import torch.nn.functional as F

# È presupposto che le classi (DifferentiableMPCController,
# GeneralQuadCost, BlockKoopmanNet) siano già definite nell'ambiente.

# --------------------------------------------------------------
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] usare dispositivo: {DEVICE}")

# ------------------------ parametri globali -------------------
BATCH_SIZE, DT, HORIZON, N_SIM = 10, 0.05, 30, 150
nx, nu = 4, 1                    # stato e controllo
# ------------------------ dinamica reale ----------------------
@dataclass(frozen=True)
class CartPoleParams:
    m_c: float; m_p: float; l: float; g: float
    @classmethod
    def from_gym(cls):
        with open(os.devnull,'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(env.unwrapped.masscart, env.unwrapped.masspole,
                   env.unwrapped.length, env.unwrapped.gravity)

def f_cartpole_true(x: torch.Tensor, u: torch.Tensor,
                    dt: float, p: CartPoleParams) -> torch.Tensor:
    pos, vel, th, om = torch.unbind(x,-1)
    force = u.squeeze(-1)
    sin, cos = torch.sin(th), torch.cos(th)
    tot_m, m_pl = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_pl * om**2 * sin) / tot_m
    th_dd = (p.g*sin - cos*temp)/(p.l*(4/3 - p.m_p*cos**2/tot_m))
    vel_dd = temp - m_pl*th_dd*cos / tot_m
    return torch.stack((pos+vel*dt,
                        vel+vel_dd*dt,
                        th +om*dt,
                        om +th_dd*dt), -1)

p_phys = CartPoleParams.from_gym()
f_true = lambda x,u,dt: f_cartpole_true(x,u,dt,p_phys)

# ==================== COSTO comune ============================
x_target = torch.tensor([-2.,0.,0.,0.], device=DEVICE)
Q_diag   = torch.tensor([10.,1.,100.,1.], device=DEVICE)
R_val    = 0.1
C_step   = torch.zeros(nx+nu,nx+nu, device=DEVICE)
C_step[:nx,:nx] = torch.diag(Q_diag)
C_step[nx:,nx:] = torch.diag(torch.tensor([R_val], device=DEVICE))
C_run    = C_step.repeat(HORIZON,1,1)
c_run    = torch.zeros(HORIZON,nx+nu, device=DEVICE)
C_final  = torch.zeros(nx+nu,nx+nu, device=DEVICE)
C_final[:nx,:nx] = torch.diag(Q_diag*10)

cost_mod = GeneralQuadCost(
    nx=nx, nu=nu, C=C_run, c=c_run, C_final=C_final,
    c_final=torch.zeros(nx+nu, device=DEVICE), device=str(DEVICE),
    x_ref=x_target.repeat(HORIZON+1,1),
    u_ref=torch.zeros(HORIZON,nu, device=DEVICE)
)

# ==============================================================
# 1) PRE-TRAINING KOOPMAN con normalizzazione + scheduler
# ==============================================================
print("\n--- 1. Pre-Training Koopman ---")
koop_net = BlockKoopmanNet(dt=DT, x_dim=nx, u_dim=nu,
                           z_dim=8, h_dim=128, aux_dim=64,
                           device=str(DEVICE)).to(DEVICE)

N_TRAIN = 500
torch.manual_seed(0)
x_train = (torch.rand(N_TRAIN,nx,device=DEVICE)-.5)*4
u_train = (torch.rand(N_TRAIN,nu,device=DEVICE)-.5)*44

with torch.no_grad():
    y_train = f_true(x_train, u_train, DT)

x_mu, x_std = x_train.mean(0), x_train.std(0)+1e-6
u_mu, u_std = u_train.mean(0), u_train.std(0)+1e-6
norm_x = lambda x: (x - x_mu)/x_std
norm_u = lambda u: (u - u_mu)/u_std

def dyn_norm(x,u,dt): return koop_net.dynamics(norm_x(x), norm_u(u), dt)
def jac_norm(x,u):     return koop_net.jacobians(norm_x(x), norm_u(u))

optimizer  = torch.optim.AdamW(koop_net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2)

# ================== MODIFICA CHIAVE ==================
EPOCHS     = 50  # <-- AUMENTATO DA 5 A 50
# =====================================================

koop_net.train()
for ep in trange(EPOCHS, desc="Koopman pre-train"):
    # Qui si potrebbe usare un DataLoader per gestire batch più piccoli,
    # ma per semplicità usiamo l'intero dataset.
    optimizer.zero_grad()
    pred = dyn_norm(x_train, u_train, DT)
    loss = F.mse_loss(pred, y_train)
    loss.backward(); optimizer.step(); scheduler.step()
koop_net.eval()

# ================== NUOVO: CONTROLLO SANITÀ JACOBIANI ==================
print("\n--- Controllo Sanità Jacobiani Post-Training ---")
with torch.no_grad():
    # Prendi un piccolo batch di dati di test
    x_sample = torch.randn(100, nx, device=DEVICE)
    u_sample = torch.randn(100, nu, device=DEVICE)
    # Calcola i Jacobiani rispetto agli input normalizzati
    A_norm_sample, B_norm_sample = jac_norm(x_sample, u_sample)
    # Stampa il valore assoluto medio di B_norm. Se > 0, il controllo ha un effetto.
    mean_abs_b_norm = torch.mean(torch.abs(B_norm_sample)).item()
    print(f"Valore assoluto medio di B_norm: {mean_abs_b_norm:.6f}")
    if mean_abs_b_norm < 1e-4:
        print("⚠️ ATTENZIONE: Il Jacobiano del controllo B_norm è quasi zero. Il modello potrebbe non aver appreso l'effetto del controllo.")
    else:
        print("✅ OK: Il modello sembra aver appreso l'effetto del controllo.")
# =======================================================================


# ==============================================================
# 2) CONTROLLER K-MPC
# ==============================================================
print("\n--- 2. Creazione dei Controller ---")
def f_dyn_k(x,u,dt):
    was_unbatched = x.ndim == 1
    if was_unbatched:
        x, u = x.unsqueeze(0), u.unsqueeze(0)
    x_next = dyn_norm(x, u, dt)
    return x_next.squeeze(0) if was_unbatched else x_next

def f_dyn_k_jac(x, u, dt):
    was_unbatched = x.ndim == 1
    if was_unbatched:
        x, u = x.unsqueeze(0), u.unsqueeze(0)
    A_norm, B_norm = jac_norm(x, u)
    # Applica la regola della catena
    A = A_norm / x_std
    B = B_norm / u_std
    A, B = A.clamp(-20, 20), B.clamp(-20, 20)
    return (A.squeeze(0), B.squeeze(0)) if was_unbatched else (A, B)

controller_kmpc = DifferentiableMPCController(
    f_dyn=f_dyn_k,
    f_dyn_jac=f_dyn_k_jac,
    grad_method="analytic",
    cost_module=cost_mod,
    step_size=DT, horizon=HORIZON, N_sim=N_SIM,
    u_min=torch.tensor([-10.], device=DEVICE),
    u_max=torch.tensor([ 10.], device=DEVICE),
    device=str(DEVICE),
    total_time=HORIZON*DT
)

# ==============================================================
# 3) CONTROLLER N-MPC (fisico)
# ==============================================================
controller_nmpc = DifferentiableMPCController(
    f_dyn=f_true, f_dyn_jac=None, grad_method="finite_diff",
    cost_module=cost_mod,
    step_size=DT, horizon=HORIZON, N_sim=N_SIM,
    u_min=torch.tensor([-10.], device=DEVICE),
    u_max=torch.tensor([ 10.], device=DEVICE),
    device=str(DEVICE),
    total_time=HORIZON*DT
)

# ==============================================================
# 4) SIMULAZIONE batch
# ==============================================================
print("\n--- 3. Esecuzione Simulazione Batch ---")
torch.manual_seed(123)
x0_base = torch.tensor([0.,0.,0.4,0.], device=DEVICE)
x0_batch = x0_base + torch.randn(BATCH_SIZE,nx,device=DEVICE)*torch.tensor([0.5,0.5,0.2,0.2], device=DEVICE)

def rollout(ctrl,name):
    print(f"[{name}] rollout …")
    tic = time.perf_counter()
    with torch.no_grad():
        X,U = ctrl.forward(x0_batch)
    if DEVICE.type=='cuda': torch.cuda.synchronize()
    print(f"    done in {(time.perf_counter()-tic)*1e3:.1f} ms")
    return X,U

Xk,Uk = rollout(controller_kmpc,"K-MPC")
Xn,Un = rollout(controller_nmpc,"N-MPC")

# ==============================================================
# 5) RISULTATI E GRAFICI (invariato)
# ==============================================================
def metrics(X,U):
    final_err = torch.linalg.norm(X[:,-1]-x_target,dim=1).mean().item()
    ctrl_eff  = (U.pow(2)).mean().item()
    return final_err, ctrl_eff

ek,uk = metrics(Xk,Uk)
en,un = metrics(Xn,Un)

print("\n--- 4. Risultati e Grafici ---")
print("\n=== Confronto Performance ===")
print(f"{'Metrica':<20} | {'K-MPC':<10} | {'N-MPC':<10}")
print("-" * 45)
print(f"{'Errore Finale (L2)':<20} | {ek:<10.3f} | {en:<10.3f}")
print(f"{'Controllo Medio (u²)':<20} | {uk:<10.3f} | {un:<10.3f}")
print("-" * 45)

plt.style.use('seaborn-v0_8-whitegrid')

Xk_mean, Xk_std = Xk.mean(0).cpu(), Xk.std(0).cpu()
Uk_mean, Uk_std = Uk.mean(0).cpu(), Uk.std(0).cpu()
Xn_mean, Xn_std = Xn.mean(0).cpu(), Xn.std(0).cpu()
Un_mean, Un_std = Un.mean(0).cpu(), Un.std(0).cpu()

time_state = torch.arange(N_SIM + 1) * DT
time_ctrl = torch.arange(N_SIM) * DT

# --- Grafico 1: Traiettorie di Stato (Media e Dev. Std.) ---
state_labels = ["Posizione x [m]", "Velocità v [m/s]", "Angolo θ [rad]", "Velocità Ang. ω [rad/s]"]
fig, axs = plt.subplots(nx, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Confronto Traiettorie Medie di Stato (±1 Dev. Std.)", fontsize=16)

for i in range(nx):
    axs[i].plot(time_state, Xk_mean[:, i], color='#0077BB', label='K-MPC Media')
    axs[i].fill_between(time_state, Xk_mean[:, i] - Xk_std[:, i], Xk_mean[:, i] + Xk_std[:, i],
                        color='#0077BB', alpha=0.2)
    axs[i].plot(time_state, Xn_mean[:, i], color='#CC3311', linestyle='--', label='N-MPC Media')
    axs[i].fill_between(time_state, Xn_mean[:, i] - Xn_std[:, i], Xn_mean[:, i] + Xn_std[:, i],
                        color='#CC3311', alpha=0.2)
    axs[i].axhline(x_target[i].cpu(), color='black', linestyle=':', linewidth=2, label='Target')
    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=3)
axs[-1].set_xlabel("Tempo [s]")
fig.tight_layout(rect=[0, 0, 1, 0.96])

# --- Grafico 2: Input di Controllo (Media e Dev. Std.) ---
plt.figure(figsize=(14, 5))
plt.title("Confronto Input di Controllo (Media ±1 Dev. Std.)", fontsize=14)
plt.plot(time_ctrl, Uk_mean, color='#0077BB', label='K-MPC Media')
plt.fill_between(time_ctrl, (Uk_mean - Uk_std).squeeze(), (Uk_mean + Uk_std).squeeze(),
                 color='#0077BB', alpha=0.2)
plt.plot(time_ctrl, Un_mean, color='#CC3311', linestyle='--', label='N-MPC Media')
plt.fill_between(time_ctrl, (Un_mean - Un_std).squeeze(), (Un_mean + Un_std).squeeze(),
                 color='#CC3311', alpha=0.2)
plt.xlabel("Tempo [s]")
plt.ylabel("Forza [N]")
plt.grid(True)
plt.legend()

# --- Grafico 3: Errore Quadratico Medio (MSE) nel tempo ---
error_k = Xk - x_target
mse_k = torch.mean(error_k**2, dim=0).cpu()
error_n = Xn - x_target
mse_n = torch.mean(error_n**2, dim=0).cpu()

plt.figure(figsize=(14, 6))
plt.title("Errore Quadratico Medio di Regolazione (Log Scale)", fontsize=14)
for i in range(nx):
    plt.plot(time_state, mse_k[:, i], color=f'C{i}', linestyle='-',
             label=f'MSE {state_labels[i]} (K-MPC)')
    plt.plot(time_state, mse_n[:, i], color=f'C{i}', linestyle='--',
             label=f'MSE {state_labels[i]} (N-MPC)')

plt.xlabel("Tempo [s]")
plt.ylabel("Errore Quadratico Medio (MSE)")
plt.yscale('log')
plt.grid(True, which="both", ls="-")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()