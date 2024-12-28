import numpy as np
from scipy.optimize import brentq

def potential(f_prime, z):
    return -np.log(np.abs(f_prime(z)))

def compute_pressure(f, f_prime, z0, t_min, t_max, delta_t, N):
    T = np.arange(t_min, t_max + delta_t, delta_t)
    P = []

    for t in T:
        S = 0
        z = z0
        for n in range(1, N + 1):
            phi_n = 0
            z_current = z
            for _ in range(n):
                phi_n += potential(f_prime, z_current)
                z_current = f(z_current)
            S += np.exp(-t * phi_n)
        P_t = (1 / N) * np.log(S)
        P.append(P_t)

    T = np.array(T)
    P = np.array(P)

    sign_changes = np.where(np.diff(np.sign(P)))[0]
    if len(sign_changes) > 0:
        t_approx = []
        for idx in sign_changes:
            t_root = brentq(lambda t: (1 / N) * np.log(
                sum(np.exp(-t * sum(potential(f_prime, f(z0 if i == 0 else f(z0)))) for i in range(1, N + 1)))), T[idx], T[idx + 1])
            t_approx.append(t_root)
        return t_approx
    else:
        return None

if __name__ == "__main__":
    def f(z):
        return z**2 + 1

    def f_prime(z):
        return 2 * z

    z0 = 0.1
    t_min, t_max = 0, 2
    delta_t = 0.1
    N = 100

    roots = compute_pressure(f, f_prime, z0, t_min, t_max, delta_t, N)
    if roots:
        print(f"Roots found: {roots}")
    else:
        print("No roots found.")
