# ---------------------------------------------------------
# geometry.py
# Géométrie adaptative à couplage spectral et de phase
# Cœur vivant du Riemann Organ
# ---------------------------------------------------------
import numpy as np

def clip(x, L=10.0):
    return np.clip(x, -L, L)

class AdaptiveGeometry:
    """
    Géométrie vivante et évolutive du réseau d'oscillateurs.
    Combine :
    - Synchronisation de phase (Kuramoto-like)
    - Proximité spectrale (zéros proches → couplage fort)
    - Mémoire et plasticité
    """
    def __init__(self, N, seed=0, base_strength=0.2):
        rng = np.random.default_rng(seed)
        self.K = base_strength * rng.random((N, N))
        np.fill_diagonal(self.K, 0)
        self.N = N

    def update(self, a, omega,
               eta=0.07,      # force de synchronisation de phase
               xi=0.03,       # force spectrale (zéros proches)
               decay=0.96,    # oubli lent
               noise=0.001):
        phases = np.angle(a)
        # Différences de phase (anti-symétrique)
        diff_phase = phases[:, None] - phases[None, :]
        # Différences fréquentielles (zéros proches dans le spectre)
        diff_omega = omega[:, None] - omega[None, :]

        # Terme de phase (Hebbien attractif)
        phase_term = eta * np.sin(diff_phase)  # sin pour synchronisation

        # Terme spectral : zéros très proches → couplage renforcé
        spectral_term = xi * np.exp(-5.0 * diff_omega**2)  # pic très fin

        # Mise à jour avec mémoire + bruit léger pour exploration
        dK = - (1.0 - decay) * self.K + phase_term + spectral_term
        dK += noise * np.random.randn(*dK.shape)

        self.K += dK
        np.fill_diagonal(self.K, 0)
        self.K = clip(self.K, L=6.0)

        # Rend le couplage symétrique (physique)
        self.K = (self.K + self.K.T) / 2.0

        return self.K
