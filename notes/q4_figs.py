"""Figures for the Q4 bridging note: the four local N's, and one global phi."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BLUE, ORANGE = "#1f4e9c", "#b5651d"

def N(i, r, s):
    return [(1-r)*(1-s)/4, (1+r)*(1-s)/4, (1+r)*(1+s)/4, (1-r)*(1+s)/4][i]

# ---------- figure 1: the four local shape functions ----------
r = np.linspace(-1, 1, 61); s = np.linspace(-1, 1, 61)
R, S = np.meshgrid(r, s)
corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
fig = plt.figure(figsize=(9.4, 2.7))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, projection="3d")
    ax.plot_surface(R, S, N(i, R, S), cmap="Blues", vmin=-0.3, vmax=1.1,
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.scatter(*corners[i], 1, color=ORANGE, s=22, depthshade=False)
    for j, c in enumerate(corners):
        if j != i:
            ax.scatter(*c, 0, color="0.45", s=10, depthshade=False)
    ax.set_title(r"$N_%d$" % (i+1), fontsize=11, pad=0)
    ax.set_zlim(0, 1); ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1])
    ax.set_zticks([0, 1]); ax.tick_params(labelsize=7)
    ax.set_xlabel("$r$", fontsize=8, labelpad=-8)
    ax.set_ylabel("$s$", fontsize=8, labelpad=-8)
    ax.view_init(26, -58)
fig.tight_layout()
fig.savefig("q4_shapes.pdf"); print("wrote q4_shapes.pdf")

# ---------- figure 2: one global basis function over four elements ----------
fig = plt.figure(figsize=(9.4, 3.4))

# (a) the 2x2 patch, flat, with node + element numbering
ax = fig.add_subplot(1, 2, 1)
for k in (0, 1, 2):
    ax.plot([0, 2], [k, k], color="0.35", lw=1.4)
    ax.plot([k, k], [0, 2], color="0.35", lw=1.4)

# element numbers, tucked into the outer corner of each quadrant
for (ex, ey, lab) in [(0.30, 0.30, "$E_1$"), (1.70, 0.30, "$E_2$"),
                      (0.30, 1.70, "$E_3$"), (1.70, 1.70, "$E_4$")]:
    ax.text(ex, ey, lab, ha="center", va="center", fontsize=11, color="0.5")

# which local N of each element becomes phi_j, placed next to node j
for (lx, ly, lab) in [(0.74, 0.74, "$N_3$"), (1.26, 0.74, "$N_4$"),
                      (0.74, 1.26, "$N_2$"), (1.26, 1.26, "$N_1$")]:
    ax.text(lx, ly, lab, ha="center", va="center", fontsize=11, color=BLUE)

# global node numbers 1..9, row-major from the bottom left
n = 1
for iy in (0, 1, 2):
    for ix in (0, 1, 2):
        big = (ix == 1 and iy == 1)
        ax.plot(ix, iy, "o", ms=16, mfc="white",
                mec=ORANGE if big else "0.35", mew=1.8 if big else 1.2, zorder=3)
        ax.text(ix, iy, str(n), ha="center", va="center", zorder=4,
                fontsize=9, color=ORANGE if big else "0.25",
                fontweight="bold" if big else "normal")
        n += 1
ax.set_xlim(-0.25, 2.25); ax.set_ylim(-0.3, 2.3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"$\varphi_5$ is four local $N$'s glued at node 5", fontsize=10)

# (b) the resulting pyramid
ax = fig.add_subplot(1, 2, 2, projection="3d")
gx = np.linspace(0, 2, 81); gy = np.linspace(0, 2, 81)
GX, GY = np.meshgrid(gx, gy)
PHI = np.clip(1-np.abs(GX-1), 0, 1)*np.clip(1-np.abs(GY-1), 0, 1)
ax.plot_surface(GX, GY, PHI, cmap="Blues", vmin=-0.3, vmax=1.1,
                linewidth=0, alpha=0.95)
ax.plot([1], [1], [1], "o", color=ORANGE, ms=6)
ax.text(1, 1, 1.12, "5", color=ORANGE, fontsize=9, ha="center")
ax.set_zlim(0, 1); ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_zticks([0, 1]); ax.tick_params(labelsize=7)
ax.set_xlabel("$x$", fontsize=8, labelpad=-6)
ax.set_ylabel("$y$", fontsize=8, labelpad=-6)
ax.view_init(24, -56)
ax.set_title(r"the global $\varphi_5$", fontsize=10)
fig.tight_layout()
fig.savefig("q4_global_phi.pdf"); print("wrote q4_global_phi.pdf")

# ---------- figure 1b: one global hat standing on a 16-element mesh ----------
NEL = 4                      # 4 x 4 elements  ->  5 x 5 = 25 nodes
JX, JY = 2, 2                # the node that carries the hat

def hat(t):
    return np.clip(1.0 - np.abs(t), 0.0, None)

fig = plt.figure(figsize=(7.2, 4.4))
ax = fig.add_subplot(111, projection="3d")

# the flat mesh on z = 0
for k in range(NEL + 1):
    ax.plot([0, NEL], [k, k], [0, 0], color="0.55", lw=1.0)
    ax.plot([k, k], [0, NEL], [0, 0], color="0.55", lw=1.0)

# all 25 nodes
for ix in range(NEL + 1):
    for iy in range(NEL + 1):
        if (ix, iy) != (JX, JY):
            ax.plot([ix], [iy], [0], "o", ms=3.2, color="0.4")

# the hat -- drawn ONLY over its support, so the rest of the mesh stays visible
g = np.linspace(JX - 1, JX + 1, 97)
GX, GY = np.meshgrid(g, g)
PHI = hat(GX - JX) * hat(GY - JY)
ax.plot_surface(GX, GY, PHI, cmap="Blues", vmin=-0.55, vmax=1.15,
                rstride=4, cstride=4, linewidth=0.25,
                edgecolors=(1, 1, 1, 0.45), antialiased=True, shade=True)

# carrying node and apex
ax.plot([JX, JX], [JY, JY], [0, 1], ls=(0, (2, 2)), color=ORANGE, lw=1.1)
ax.plot([JX], [JY], [0], "o", ms=6.5, color=ORANGE)
ax.plot([JX], [JY], [1], "o", ms=5.5, color=ORANGE)
ax.text(JX - 1.05, JY - 0.30, 1.00, r"$\varphi_j=1$ at node $j$",
        color=ORANGE, fontsize=9)

ax.set_zlim(0, 1.05)
ax.set_xticks(range(NEL + 1)); ax.set_yticks(range(NEL + 1)); ax.set_zticks([0, 1])
ax.tick_params(labelsize=7, pad=0.5)
ax.set_xlabel("$x$", fontsize=9, labelpad=0)
ax.set_ylabel("$y$", fontsize=9, labelpad=0)
ax.view_init(30, -54)
ax.set_box_aspect((1, 1, 0.58))
fig.tight_layout()
fig.savefig("q4_hat_on_mesh.pdf")
print("wrote q4_hat_on_mesh.pdf")
