"""
HW4 Problem 1 -- PyDistMesh: check that the downloaded code runs.

Runs the three examples that ship with the Canvas bundle, unchanged in their
arguments (example1.py, example2.py, poisson_example.py), headless, and writes

    hw4_p1_examples.pdf   three panels: the two example meshes and the
                          cG(1) Poisson solution of poisson_example.py
    hw4_p1_table.tex      node / element counts, mesh area against the exact
                          area, and the lecture's regularity ratio
                          max_j h_j / rho_j  (Lecture 13, CG(1) in 2D)

Nothing in the bundle is modified; this script only calls it.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse.linalg

import distmesh2d as dm
import distance_functions as distfunc
import boundary as bnd
import assemble

np.random.seed(0)          # distmesh2d thins the initial points at random


# ------------------------------------------------------------ mesh quality
def regularity(p, t, area):
    """max over elements of h_j / rho_j  (longest edge / inscribed diameter).
    Lecture 13: rho_j = diameter of the largest inscribed circle = 2*area/s,
    s = half the perimeter.  Equilateral triangle gives sqrt(3) = 1.732."""
    a = np.linalg.norm(p[t[:, 1]] - p[t[:, 0]], axis=1)
    b = np.linalg.norm(p[t[:, 2]] - p[t[:, 1]], axis=1)
    c = np.linalg.norm(p[t[:, 0]] - p[t[:, 2]], axis=1)
    h = np.max(np.vstack((a, b, c)), axis=0)
    s = 0.5 * (a + b + c)
    rho = 2.0 * area / s
    return (h / rho).max()


rows = []
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

# ------------------------------------------------- example1: unit disc
t0 = time.time()
fd1 = lambda p: np.sqrt((p**2).sum(1)) - 1.0
p, t = dm.distmesh2d(fd1, distfunc.huniform, 0.1, (-1, -1, 1, 1))
p, t, nin, nbd, area = bnd.bnd2d(fd1, p, t, -1.0, -1.0, 1.0, 1.0)
rows.append(("unit disc, uniform", 0.1, p.shape[0], nin, nbd, t.shape[0],
             area.sum(), np.pi, regularity(p, t, area), time.time() - t0))
ax = axes[0]
ax.triplot(p[:, 0], p[:, 1], t, color="k", lw=0.4)
ax.plot(p[:nin, 0], p[:nin, 1], ".", color="#1f4e9c", ms=3)
ax.plot(p[nin:, 0], p[nin:, 1], ".", color="#b5651d", ms=4)
ax.set_title("example1: unit disc, $h_0=0.1$", fontsize=10)

# -------------------------- example2: rectangle with hole, graded h
t0 = time.time()
fd2 = lambda p: distfunc.ddiff(distfunc.drectangle(p, -1, 1, -1, 1),
                               distfunc.dcircle(p, 0, 0, 0.5))
fh2 = lambda p: 0.1 + 0.3 * distfunc.dcircle(p, 0, 0, 0.5)
p, t = dm.distmesh2d(fd2, fh2, 0.04, (-1, -1, 1, 1),
                     [(-1, -1), (-1, 1), (1, -1), (1, 1)])
p, t, nin, nbd, area = bnd.bnd2d(fd2, p, t, -1.0, -1.0, 1.0, 1.0)
rows.append(("square with hole, graded", 0.04, p.shape[0], nin, nbd,
             t.shape[0], area.sum(), 4.0 - np.pi * 0.25,
             regularity(p, t, area), time.time() - t0))
ax = axes[1]
ax.triplot(p[:, 0], p[:, 1], t, color="k", lw=0.3)
ax.plot(p[:nin, 0], p[:nin, 1], ".", color="#1f4e9c", ms=2)
ax.plot(p[nin:, 0], p[nin:, 1], ".", color="#b5651d", ms=3)
ax.set_title("example2: graded towards the hole", fontsize=10)

# ---------------- poisson_example: uniform h0 = 0.04, cG(1) solve
t0 = time.time()
p, t = dm.distmesh2d(fd2, distfunc.huniform, 0.04, (-1, -1, 1, 1),
                     [(-1, -1), (-1, 1), (1, -1), (1, 1)])
p, t, nin, nbd, area = bnd.bnd2d(fd2, p, t, -1.0, -1.0, 1.0, 1.0)
A = assemble.stiffness_matrix(p, t, nin, area)
b = assemble.rhs(p, t, nin, area, lambda x, y: 1.0)
U = np.zeros(nin + nbd)
U[:nin] = scipy.sparse.linalg.spsolve(A, b)
rows.append(("square with hole, uniform, Poisson", 0.04, p.shape[0],
             nin, nbd, t.shape[0], area.sum(), 4.0 - np.pi * 0.25,
             regularity(p, t, area), time.time() - t0))
ax = axes[2]
pc = ax.tripcolor(p[:, 0], p[:, 1], t, U, shading="gouraud", cmap="inferno")
ax.tricontour(p[:, 0], p[:, 1], t, U, colors="k", linewidths=0.4, levels=8)
fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
ax.set_title("poisson_example: cG(1), $u$ on %d interior nodes" % nin,
             fontsize=10)
umax = U.max()

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig("hw4_p1_examples.pdf")

# ------------------------------------------------------------- table
with open("hw4_p1_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n")
    f.write("example & $h_0$ & interior & boundary & elements & "
            "mesh area & exact & $\\max_j h_j/\\rho_j$\\\\\n\\midrule\n")
    for (name, h0, nn, nin, nbd, ne, am, ae, reg, dt) in rows:
        f.write("%s & %.2f & %d & %d & %d & %.4f & %.4f & %.2f\\\\\n"
                % (name, h0, nin, nbd, ne, am, ae, reg))
    f.write("\\bottomrule\n\\end{tabular}\n")

for r in rows:
    print("%-42s h0=%.2f nodes=%4d int=%4d bnd=%3d elems=%4d area=%.6f "
          "exact=%.6f rel=%+.2e maxh/rho=%.3f  %.1fs"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
             (r[6] - r[7]) / r[7], r[8], r[9]))
print("poisson example: max U = %.6f at node %d, (x,y)=(%.4f,%.4f)"
      % (umax, U.argmax(), p[U.argmax(), 0], p[U.argmax(), 1]))
