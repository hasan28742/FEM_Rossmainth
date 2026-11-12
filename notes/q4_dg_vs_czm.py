"""Case 9 vs Case 10: the same machinery, two different verdicts.

Left  -- interior-penalty DG for -u'' = f on a bar. The exact answer is smooth,
         so every jump the method allows is error: refine and they vanish.
Right -- the same bar with a cohesive interface at mid-span. The exact answer
         really jumps, so the jump is the answer: refine and it stays put.

Both solvers are real; nothing in this figure is sketched by hand.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

BAD, GOOD, EXACT, ELEM = "#c62828", "#2e7d32", "0.35", "#1f6fb4"
HALO = dict(fc="white", ec="none", alpha=0.85, pad=1.0)

# ----------------------------------------------------------------- case 9: DG
uex = lambda x: x*(1-x)*np.exp(3*x)                 # smooth: no jump anywhere
fex = lambda x: np.exp(3*x)*(9*x**2 + 3*x - 4)      # f = -u''

def dg_solve(N, sigma=4.0):
    """P1 symmetric interior-penalty DG; Dirichlet imposed strongly at the ends."""
    h = 1.0/N; nd = 2*N
    K = np.zeros((nd, nd)); F = np.zeros(nd)
    gp, gw = np.polynomial.legendre.leggauss(8)
    for e in range(N):
        i, j = 2*e, 2*e+1
        K[np.ix_([i, j], [i, j])] += np.array([[1, -1], [-1, 1]])/h
        for xi, w in zip(gp, gw):
            t = (xi+1)/2; f = fex(e*h + t*h)
            F[i] += f*(1-t)*w*h/2; F[j] += f*t*w*h/2
    for e in range(N-1):                                    # interior faces
        J = np.zeros(nd); A = np.zeros(nd)
        J[2*e+1] = 1.0; J[2*e+2] = -1.0                     # [u] = u^- - u^+
        A[2*e+1] += 0.5/h; A[2*e]   -= 0.5/h                # {u'}
        A[2*e+3] += 0.5/h; A[2*e+2] -= 0.5/h
        K += -np.outer(A, J) - np.outer(J, A) + (sigma/h)*np.outer(J, J)
    free = [k for k in range(nd) if k not in (0, nd-1)]
    U = np.zeros(nd)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    seg = [((e*h, (e+1)*h), (U[2*e], U[2*e+1])) for e in range(N)]
    jump = np.array([U[2*e+1] - U[2*e+2] for e in range(N-1)])
    return seg, jump

# --------------------------------------------------------------- case 10: CZM
Tmax, df, Uend = 1.0, 0.1, 0.64        # T(d) = Tmax(1 - d/df); end displacement

def czm_solve(n):
    """Bar with a doubled node at x=1/2 carrying a traction-separation law."""
    h = 0.5/n; nn = 2*(n+1)
    K = np.zeros((nn, nn)); F = np.zeros(nn)
    for e in range(n):
        K[np.ix_([e, e+1], [e, e+1])] += np.array([[1, -1], [-1, 1]])/h
        i = n+1+e
        K[np.ix_([i, i+1], [i, i+1])] += np.array([[1, -1], [-1, 1]])/h
    kc = -Tmax/df                                   # the interface is in softening
    K[np.ix_([n, n+1], [n, n+1])] += np.array([[kc, -kc], [-kc, kc]])
    F[n] += Tmax; F[n+1] -= Tmax                    # traction resists opening
    fixed = {0: 0.0, nn-1: Uend}
    free = [k for k in range(nn) if k not in fixed]
    rhs = F[free] - K[np.ix_(free, list(fixed))] @ np.array(list(fixed.values()))
    U = np.zeros(nn)
    for k, v in fixed.items(): U[k] = v
    U[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
    xl = np.linspace(0, 0.5, n+1); xr = np.linspace(0.5, 1, n+1)
    return (xl, U[:n+1]), (xr, U[n+1:]), U[n+1] - U[n]

# ===================================================================== figure
fig = plt.figure(figsize=(10.4, 9.5))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.80, 1.0, 0.78],
              hspace=0.70, wspace=0.22, left=0.085, right=0.975,
              top=0.885, bottom=0.165)

# ------------------------------------------------- row 1: the mesh / the nodes
def meshstrip(ax, broken, title, note, col):
    """broken = set of interior face indices whose nodes are doubled."""
    NE, g = 4, 0.016
    for e in range(NE):
        a, b = e/NE, (e+1)/NE
        a += g if (e in broken or e == 0) else 0.0
        b -= g if ((e+1) in broken or e == NE) else 0.0
        ax.add_patch(plt.Rectangle((a, -0.16), b-a, 0.32, fc="#eaf0f7",
                                   ec=ELEM, lw=1.8, zorder=2))
    for k in range(1, NE):
        x = k/NE
        if k in broken:                                   # two values, may differ
            for dx in (-g, +g):
                ax.plot(x+dx, 0, "o", ms=9, mfc="white", mec=col, mew=2.2, zorder=4)
        else:                                             # one value, shared
            ax.plot(x, 0, "o", ms=9, color="#15284a", zorder=4)
    ax.plot(0-g, 0, "o", ms=9, color="#15284a", zorder=4)
    ax.plot(1+g, 0, "o", ms=9, color="#15284a", zorder=4)
    ax.text(0.5, -2.25, note, ha="center", va="center", fontsize=10.5, color=col)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-2.65, 0.5); ax.axis("off")
    return ax

axM1 = meshstrip(fig.add_subplot(gs[0, 0]), {1, 2, 3},
                 "Case 9 --- DG for $-u''=f$ in a bar\nthe exact answer is $\\mathit{smooth}$",
                 "every face broken: each element owns its own nodes", BAD)
axM2 = meshstrip(fig.add_subplot(gs[0, 1]), {2},
                 "Case 10 --- cohesive zone in the same bar\n"
                 "the exact answer $\\mathit{really\\ jumps}$",
                 "only the crack path broken; elsewhere the node is shared", GOOD)
axM1.annotate("$\\circ\\,\\circ$ two values, free to differ\n$\\rightarrow$ DISCONTINUOUS here",
              (0.5, -0.17), textcoords="offset points", xytext=(0, -24), ha="center",
              va="top", fontsize=10, color=BAD,
              arrowprops=dict(arrowstyle="-", lw=0.9, color=BAD))
axM2.annotate("$\\bullet$ one shared value\n$\\rightarrow$ CONTINUOUS here",
              (0.25, -0.17), textcoords="offset points", xytext=(-40, -24), ha="center",
              va="top", fontsize=10, color="#15284a",
              arrowprops=dict(arrowstyle="-", lw=0.9, color="#15284a"))
axM2.annotate("$\\circ\\,\\circ$ two values\n$\\rightarrow$ DISCONTINUOUS", (0.5, -0.17),
              textcoords="offset points", xytext=(50, -24), ha="center", va="top",
              fontsize=10, color=GOOD,
              arrowprops=dict(arrowstyle="-", lw=0.9, color=GOOD))

# ------------------------------------------------------ row 2: the solutions
axA = fig.add_subplot(gs[1, 0])
xx = np.linspace(0, 1, 400)
axA.plot(xx, uex(xx), ls=(0, (6, 3)), lw=2.0, color=EXACT, label="exact (no jump)")
seg, jump = dg_solve(4)
for k, ((a, b), (ua, ub)) in enumerate(seg):
    axA.plot([a, b], [ua, ub], color=ELEM, lw=2.6, zorder=3,
             label="DG solution, 4 elements" if k == 0 else None)
    axA.plot([a, b], [ua, ub], "o", ms=5, mfc="white", mec=ELEM, mew=1.6, zorder=4)
kmax = int(np.argmax(np.abs(jump))); xf = (kmax+1)/4
lo, hi = sorted([seg[kmax][1][1], seg[kmax+1][1][0]])
axA.add_patch(FancyArrowPatch((xf, lo), (xf, hi), arrowstyle="<|-|>",
                              mutation_scale=11, lw=1.6, color=BAD, zorder=6))
axA.annotate(f"gap $={abs(jump[kmax]):.3f}$\nthis is $\\mathbf{{error}}$",
             (xf, (lo+hi)/2), textcoords="offset points", xytext=(26, -52),
             ha="left", fontsize=11, color=BAD, bbox=HALO, zorder=7)
axA.legend(fontsize=10, loc="upper left", frameon=False)

axB = fig.add_subplot(gs[1, 1])
(xl, ul), (xr, ur), d = czm_solve(4)
axB.plot([0, 0.5], [0, 0.6*0.5], ls=(0, (6, 3)), lw=2.0, color=EXACT,
         label="exact (jumps at mid-span)")
axB.plot([0.5, 1], [0.6*0.5+d, Uend], ls=(0, (6, 3)), lw=2.0, color=EXACT)
axB.plot(xl, ul, color=ELEM, lw=2.6, zorder=3, label="FE solution, 4+4 elements")
axB.plot(xr, ur, color=ELEM, lw=2.6, zorder=3)
axB.plot(np.r_[xl, xr], np.r_[ul, ur], "o", ms=5, mfc="white", mec=ELEM, mew=1.6, zorder=4)
axB.add_patch(FancyArrowPatch((0.5, ul[-1]), (0.5, ur[0]), arrowstyle="<|-|>",
                              mutation_scale=11, lw=1.6, color=GOOD, zorder=6))
axB.annotate(f"gap $={d:.4f}$\nthis is the $\\mathbf{{crack\\ opening}}$",
             (0.5, (ul[-1]+ur[0])/2), textcoords="offset points", xytext=(16, -30),
             ha="left", fontsize=11, color=GOOD, bbox=HALO, zorder=7)
axB.legend(fontsize=10, loc="upper left", frameon=False)

for ax, ttl in [(axA, "the four element solutions do not meet"),
                (axB, "the two halves do not meet --- and should not")]:
    ax.set_xlabel("$x$", fontsize=12); ax.set_ylabel("$u(x)$", fontsize=12)
    ax.set_title(ttl, fontsize=11, color="0.3", pad=6)
    ax.tick_params(labelsize=10); ax.spines[["top", "right"]].set_visible(False)

# --------------------------------------------- row 3: what refinement does
NS = [4, 8, 16, 32, 64]
axC = fig.add_subplot(gs[2, 0])
gaps = [np.abs(dg_solve(N)[1]).max() for N in NS]
axC.loglog(NS, gaps, "o-", color=BAD, lw=2.0, ms=7)
axC.annotate("the gap collapses --- it was\nnever anything but error",
             (NS[1], gaps[1]), textcoords="offset points", xytext=(16, 22),
             fontsize=10.5, color=BAD)
axC.set_ylabel("largest gap", fontsize=12)

axD = fig.add_subplot(gs[2, 1])
ds = [czm_solve(N)[2] for N in NS]
axD.semilogx(NS, ds, "o-", color=GOOD, lw=2.0, ms=7)
axD.set_ylim(0.0, 0.08)
axD.annotate(f"${ds[0]:.4f}$ on every mesh --- a physical\n"
             "answer, not a discretisation error",
             (NS[1], ds[1]), textcoords="offset points", xytext=(-24, 26),
             fontsize=10.5, color=GOOD)
axD.set_ylabel("crack opening", fontsize=12)

for ax in (axC, axD):
    ax.set_xlabel("elements", fontsize=12)
    ax.tick_params(labelsize=10); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])

# ------------------------------------------------------------- the verdicts
for xc, col, txt in [
    (0.278, BAD,
     "DG  $\\Rightarrow$  NON-CONFORMING\n"
     "the face term is a penalty you tune; it exists only to\n"
     "suppress a jump the true answer never had"),
    (0.755, GOOD,
     "DG machinery  $\\Rightarrow$  CONFORMING\n"
     "the face term is a traction--separation law with a real $G_c$;\n"
     "the jump it allows is the one the true answer has")]:
    fig.text(xc, 0.018, txt, ha="center", va="bottom", fontsize=11,
             color=col, linespacing=1.5,
             bbox=dict(fc="white", ec=col, lw=1.3, pad=5.5))

fig.savefig("q4_dg_vs_czm.pdf")
print("wrote q4_dg_vs_czm.pdf")
print(f"  DG  largest gap by mesh: {[f'{g:.3e}' for g in gaps]}")
print(f"  CZM crack opening     : {[f'{v:.6f}' for v in ds]}")
