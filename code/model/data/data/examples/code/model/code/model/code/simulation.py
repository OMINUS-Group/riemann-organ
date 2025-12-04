# ---------------------------------------------------------
# simulation.py
# Simulation principale du Riemann Organ
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from model.oscillator import OscillatorSystem
from model.geometry import AdaptiveGeometry
from utils.normalize_zeros import load_zeros, normalize_gammas

def simulate(gammas_path="../data/odlyzko_zeros_100.txt",
             T=250.0, dt=0.01, lam=0.6):
    gamma = load_zeros(gammas_path)
    omega = normalize_gammas(gamma, alpha=1.2)

    osc = OscillatorSystem(omega, lam=lam, dt=dt)
    geo = AdaptiveGeometry(N=len(omega))

    steps = int(T / dt)
    order_param = np.zeros(steps)

    for k in range(steps):
        if k % 1000 == 0:
            print(f"Step {k}/{steps}")

        K = geo.update(osc.a, osc.omega)
        osc.step(K)
        osc.update_energy()
        osc.update_metric()

        order_param[k] = np.abs(np.mean(osc.a / (np.abs(osc.a) + 1e-12)))

    return order_param

# === Lancement direct ===
if __name__ == "__main__":
    R = simulate("../data/odlyzko_zeros_100.txt", T=200)
    plt.figure(figsize=(14, 5))
    plt.plot(R, color='#4B0082', lw=1.8)
    plt.title("Riemann Organ — 100 premiers zéros non triviaux", fontsize=18)
    plt.xlabel("temps")
    plt.ylabel("paramètre d’ordre |R(t)|")
    plt.grid(alpha=0.3)
    plt.show()
