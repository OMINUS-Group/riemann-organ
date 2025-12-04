# ---------------------------------------------------------
# oscillator.py
# Oscillateur Stuart–Landau généralisé avec champ interne
# ---------------------------------------------------------
import numpy as np

def clip(x, L=10.0):
    return np.clip(x, -L, L)

class OscillatorSystem:
    def __init__(self, omega, lam=0.6, dt=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.N = len(omega)
        self.dt = dt
        self.lam = lam
        self.omega = omega.astype(complex)
        self.a = 0.08 * (rng.normal(size=self.N) + 1j * rng.normal(size=self.N))
        self.E = np.zeros(self.N)   # énergie interne
        self.g = np.zeros(self.N)   # métrique interne (tension)

    def step(self, K):
        a, omega, dt = self.a, self.omega, self.dt

        # Ordre global (normalisé)
        O = a / (np.abs(a) + 1e-12)
        mean_O = np.mean(O)

        # Feedback non-linéaire borné (champ cohérent global → local)
        F = 0.08 * clip(self.g * a) \
            + 0.06 * clip(self.E * a) \
            + 0.05 * clip(mean_O * np.ones_like(a))  # réinjection globale

        # Couplage diffusif via géométrie adaptative
        coupling = K @ a - np.sum(K, axis=1) * a

        # Équation principale
        da = (self.lam + 1j * omega) * a \
             - (1.0 + 1j * 0.1) * np.abs(a)**2 * a \
             + coupling + F

        self.a = a + dt * da

    def update_energy(self, gamma=0.08, delta=0.06):
        dE = -gamma * self.E + delta * np.abs(self.a)**2
        self.E = clip(self.E + self.dt * dE, L=12)

    def update_metric(self, rho=0.04):
        dg = rho * clip(np.real(self.a * np.conj(self.a.mean())))
        self.g = clip(self.g + self.dt * dg, L=4)
