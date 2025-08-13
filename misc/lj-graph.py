import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def function1():
    sigma = 3
    epsilon = 10
    rmin = 2**(1/6)
    dist = np.linspace(0.8/sigma,2.0/sigma,1000)
    lj_det = np.zeros(1000)

    for idx, r in enumerate(dist):
        inv_r = sigma / r
        #inv_r6 = (sigma * inv_r) ** 6
        #inv_r12 = inv_r6 ** 2
        #lj_force = 24 * epsilon * (2 * inv_r12 - inv_r6) / inv_r
        lj_pot  = 4 *  (inv_r**12 - inv_r**6)
        lj_det[idx] = lj_pot / epsilon

    #plt.figure(figsize=(8,5))
    #plt.plot(dist, lj_det , lw=2, label="Force $F/(\epsilon/\\sigma)$")
    #plt.axhline(0, ls=':', alpha=0.8)
    #plt.axvline(rmin, ls='--', alpha=0.8, label=f"$r_\\min/\\sigma = {rmin:.3f}$")
    #plt.fill_between(dist, lj_det, 0, where=(lj_det<0), alpha=0.15, label="Attractive region")
    #plt.xlabel("$r/\\sigma$")
    #plt.ylabel("$F\\, /\\, (\\epsilon/\\sigma)$")
    #plt.title("Lennard–Jones Force (reduced units)")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(dist, lj_det, lw=2, label="Potential $V/\\epsilon$")
    plt.axhline(-1, ls=':', alpha=0.8, label="Well depth $V/\\epsilon=-1$")
    plt.axvline(rmin, ls='--', alpha=0.8, label=f"$r_\\min/\\sigma = {rmin:.3f}$")
    plt.xlabel("$r/\\sigma$")
    plt.ylabel("$V\\, /\\, \\epsilon$")
    plt.title("Lennard–Jones Potential (reduced units)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def function2():
        # Parameters
    sigma = 10.0
    epsilon = 10.0

    # Reduced coordinate r* = r/sigma
    rstar = np.linspace(0.8, 4.0, 1000)           # covers repulsive + attractive
    inv = 1.0 / rstar

    # Reduced potential and force (dimensionless)
    Vstar = 4 * (inv**12 - inv**6)                # V / epsilon
    Fstar = 24 * (2*inv**12 - inv**6) / rstar     # F / (epsilon/sigma)

    # Physical r and scalings if you prefer dimensional axes/values
    r = rstar * sigma
    V = epsilon * Vstar
    F = (epsilon / sigma) * Fstar

    rmin = 2**(1/6)        # in reduced units
    rmin_dim = rmin * sigma

    # --- Plot 1: Force (dimensionless) ---
    plt.figure(figsize=(8,5))
    plt.plot(rstar, Fstar, lw=2, label="Force $F/(\epsilon/\\sigma)$")
    plt.axhline(0, ls=':', alpha=0.8)
    plt.axvline(rmin, ls='--', alpha=0.8, label=f"$r_\\min/\\sigma = {rmin:.3f}$")
    plt.fill_between(rstar, Fstar, 0, where=(Fstar<0), alpha=0.15, label="Attractive region")
    plt.xlabel("$r/\\sigma$")
    plt.ylabel("$F\\, /\\, (\\epsilon/\\sigma)$")
    plt.title("Lennard–Jones Force (reduced units)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Potential (dimensionless) ---
    plt.figure(figsize=(8,5))
    plt.plot(rstar, Vstar, lw=2, label="Potential $V/\\epsilon$")
    plt.axhline(-1, ls=':', alpha=0.8, label="Well depth $V/\\epsilon=-1$")
    plt.axvline(rmin, ls='--', alpha=0.8, label=f"$r_\\min/\\sigma = {rmin:.3f}$")
    plt.xlabel("$r/\\sigma$", fontsize = 20)
    plt.ylabel("$V\\, /\\, \\epsilon$", fontsize = 20)
    plt.title("Lennard–Jones Potential (reduced units)", fontsize= 20)
    plt.legend()
    plt.tight_layout()
    plt.show()


function2()


