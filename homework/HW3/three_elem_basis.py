"""Plot the six CG(3) basis functions for a 3-element cantilever mesh."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NE = 3; h = 1.0/NE; xs = np.array([i*h for i in range(NE+1)])

def phi(node, kind, x):
    j = node
    if xs[j-1] <= x <= xs[j]:
        t = (x-xs[j-1])/h; d = x-xs[j-1]
        return (3*t**2-2*t**3) if kind == 1 else (-d**2/h + d**3/h**2)
    if j+1 <= NE and xs[j] <= x <= xs[j+1]:
        s = (x-xs[j+1])/(-h); d = x-xs[j+1]
        return (3*s**2-2*s**3) if kind == 1 else (d**2/h + d**3/h**2)
    return 0.0

X = np.linspace(0, 1, 601)
fig, ax = plt.subplots(2, 3, figsize=(9.2, 4.2), sharex=True, sharey="row")
for k in (1, 2):
    for n in (1, 2, 3):
        a = ax[k-1][n-1]
        Y = [phi(n, k, x) for x in X]
        a.axhline(0, color="0.6", lw=0.8)
        for xv in xs:
            a.axvline(xv, color="0.85", lw=0.8, zorder=0)
        a.plot(X, Y, lw=2.0, color=("#1f4e9c" if k == 1 else "#b5651d"))
        a.set_title(r"$\phi_%d^{(%d)}$   (%s at node %d)"
                    % (n, k, "deflection" if k == 1 else "slope", n), fontsize=10)
        a.set_xticks(xs); a.set_xticklabels(["0", "1/3", "2/3", "1"], fontsize=8)
        a.tick_params(labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
ax[0][0].set_ylabel("value", fontsize=9)
ax[1][0].set_ylabel("value", fontsize=9)
fig.tight_layout()
fig.savefig("three_elem_basis.pdf")
print("wrote three_elem_basis.pdf")
