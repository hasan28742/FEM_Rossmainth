"""Element F = (0,0), (-2,2), (6,7), (5,-3): a valid quad listed the wrong way round.

Left  -- as listed. The traversal is clockwise, so det J < 0 at all four nodes.
Right -- the same four points relisted counter-clockwise (swap nodes 2 and 4).
         Same shape, same area; every det J flips sign.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

NODE, GAUSS, MID = "#15284a", "#55585c", "#2e7d32"
BAD, GOOD, EDGE = "#c62828", "#2e7d32", "#1f6fb4"
HALO = dict(fc="white", ec="none", alpha=0.8, pad=0.5)

CW  = [(0.0, 0.0), (-2.0, 2.0), (6.0, 7.0), (4.0, -1.0)]   # as the user listed them
CCW = [(0.0, 0.0), (4.0, -1.0), (6.0, 7.0), (-2.0, 2.0)]   # nodes 2 and 4 swapped

def shapes(r, s):
    return np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])/4.0

def detJ(P, r, s):
    X = np.array([p[0] for p in P]); Y = np.array([p[1] for p in P])
    dNr = np.array([-(1-s),  (1-s), (1+s), -(1+s)])/4.0
    dNs = np.array([-(1-r), -(1+r), (1+r),  (1-r)])/4.0
    return (dNr@X)*(dNs@Y) - (dNs@X)*(dNr@Y)

def img(P, r, s):
    n = shapes(r, s)
    return n @ np.array([p[0] for p in P]), n @ np.array([p[1] for p in P])

g  = 1/np.sqrt(3)
GP = [(-g, -g), (g, -g), (g, g), (-g, g)]
CORN = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
CEN = (2.0, 2.0)

def panel(ax, P, col, title, sub, gauss, cenoff=(0, 17)):
    for i in range(4):                                     # edges, with the traversal shown
        j = (i+1) % 4
        ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color=EDGE, lw=2.4, zorder=2)
        a = np.array(P[i]) + 0.42*(np.array(P[j]) - np.array(P[i]))
        b = np.array(P[i]) + 0.62*(np.array(P[j]) - np.array(P[i]))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=17,
                                     lw=0, color=col, zorder=5))
    for i, (px, py) in enumerate(P):                       # node badge + det J at that node
        d  = detJ(P, *CORN[i])
        dx, dy = px-CEN[0], py-CEN[1]; n = np.hypot(dx, dy)
        ax.plot(px, py, "o", ms=19, color=NODE, zorder=8)
        ax.annotate(str(i+1), (px, py), ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold", zorder=9)
        ax.annotate(f"$({px:g},{py:g})$\n$\\det J={d:g}$",
                    (px, py), textcoords="offset points", xytext=(40*dx/n, 40*dy/n),
                    ha="center", va="center", fontsize=10, color=col, zorder=9, bbox=HALO)
    if gauss:
        key = []                                   # the quad is thin: values go in a key
        for q, (r, s) in enumerate(GP, 1):
            qx, qy = img(P, r, s)
            ax.plot(qx, qy, "s", ms=16, color=GAUSS, zorder=7)
            ax.annotate(str(q), (qx, qy), ha="center", va="center", fontsize=9.5,
                        color="white", fontweight="bold", zorder=8)
            key.append(f"$q_{q}$:  {detJ(P, r, s):6.3f}")
        ax.text(0.025, 0.975, "$\\det J$ at the Gauss points\n" + "\n".join(key),
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                color=GAUSS, linespacing=1.5, zorder=9,
                bbox=dict(fc="white", ec="0.8", lw=0.8, alpha=0.95, pad=4))
    cx, cy = img(P, 0, 0)
    ax.plot(cx, cy, "D", ms=9, mfc="white", mec=MID, mew=2.2, zorder=7)
    ax.annotate(r"$(r,s)=(0,0)$", (cx, cy), textcoords="offset points", xytext=cenoff,
                ha="center", fontsize=9, color=MID, zorder=8, bbox=HALO)
    ax.set_title(title + "\n" + sub, fontsize=10.5, pad=8, color=col)
    ax.set_xlabel("$x$", fontsize=12); ax.set_ylabel("$y$", fontsize=12)
    ax.set_xlim(-6.0, 10.0); ax.set_ylim(-4.8, 11.2); ax.set_aspect("equal")
    ax.tick_params(labelsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)

fig = plt.figure(figsize=(10.6, 5.9))
gs  = GridSpec(1, 2, figure=fig, wspace=0.26, left=0.07, right=0.98,
               top=0.855, bottom=0.235)
axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
panel(axL, CW,  BAD,
      "as listed: $(0,0),(-2,2),(6,7),(4,-1)$",
      r"traversal is clockwise $\Rightarrow\ \det J<0$ at all four", gauss=False)
panel(axR, CCW, GOOD,
      "relisted: swap nodes $2$ and $4$",
      r"same shape, same area $30$; every sign flipped", gauss=True,
      cenoff=(0, 19))

axL.annotate("node 1 is a\n$149.0^\\circ$ corner", (0, 0), textcoords="offset points",
             xytext=(50, 8), fontsize=9, color="0.35", ha="center", bbox=HALO)
axR.annotate(r"node 2 is a true $90^\circ$", (4, -1), textcoords="offset points",
             xytext=(50, 30), fontsize=9, color="0.35", ha="center", bbox=HALO)

fig.text(0.5, 0.012,
         "The four points are a perfectly good convex quadrilateral --- the element is not "
         "broken, the node list is.\n"
         r"All four $\det J$ carry the $\mathit{same}$ sign, which is the signature of a "
         "reversed list, not of a tangled element (mixed signs would be that).\n"
         r"Relisted, $\det J=\frac{15}{2}+\frac{7r}{2}+\frac{5s}{2}$: corners "
         r"$\frac{3}{2},\ \frac{17}{2},\ \frac{27}{2},\ \frac{13}{2}$, so $\max/\min=9$ --- "
         "valid, but still a distorted element.",
         ha="center", va="bottom", fontsize=10, color="0.28", linespacing=1.45)
fig.savefig("q4_elemF.pdf")
print("wrote q4_elemF.pdf")
for nm, P in [("as listed (CW)", CW), ("relisted (CCW)", CCW)]:
    print(f"  {nm:16s} corners {[round(detJ(P,*c),4) for c in CORN]}  "
          f"gauss {[round(detJ(P,*q),4) for q in GP]}  sum {sum(detJ(P,*q) for q in GP):.4f}")
