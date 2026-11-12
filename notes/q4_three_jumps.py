"""When is a jump an error, and when is it the answer?

Three cases, all solved for real, none sketched:

  A  DG for -u'' = f          second-order PDE, exact answer smooth
  B  DG for a Burgers shock   first-order PDE, exact answer genuinely jumps
  C  cohesive zone in a bar   interface law, exact answer jumps at the crack

A and B use the SAME machinery -- broken space, numerical flux on every face.
Only the equation changed, and with it the verdict.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

BAD, GOOD, EXACT, ELEM = "#c62828", "#2e7d32", "0.35", "#1f6fb4"
HALO = dict(fc="white", ec="none", alpha=0.85, pad=1.0)
GL3 = np.polynomial.legendre.leggauss(3)

# =============================================== A: interior-penalty DG, -u''=f
uexA = lambda x: x*(1-x)*np.exp(3*x)
fexA = lambda x: np.exp(3*x)*(9*x**2 + 3*x - 4)

def dgA(N, sigma=4.0):
    h = 1.0/N; nd = 2*N
    K = np.zeros((nd, nd)); F = np.zeros(nd)
    gp, gw = np.polynomial.legendre.leggauss(8)
    for e in range(N):
        i, j = 2*e, 2*e+1
        K[np.ix_([i, j], [i, j])] += np.array([[1, -1], [-1, 1]])/h
        for xi, w in zip(gp, gw):
            t = (xi+1)/2; f = fexA(e*h + t*h)
            F[i] += f*(1-t)*w*h/2; F[j] += f*t*w*h/2
    for e in range(N-1):
        J = np.zeros(nd); A = np.zeros(nd)
        J[2*e+1] = 1.0; J[2*e+2] = -1.0
        A[2*e+1] += 0.5/h; A[2*e] -= 0.5/h
        A[2*e+3] += 0.5/h; A[2*e+2] -= 0.5/h
        K += -np.outer(A, J) - np.outer(J, A) + (sigma/h)*np.outer(J, J)
    free = [k for k in range(nd) if k not in (0, nd-1)]
    U = np.zeros(nd); U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    seg = [((e*h, (e+1)*h), (U[2*e], U[2*e+1])) for e in range(N)]
    return seg, np.array([U[2*e+1] - U[2*e+2] for e in range(N-1)])

# ========================================= B: DG for a stationary Burgers shock
fB = lambda u: 0.5*u*u

def godunov(uL, uR):
    s = 0.5*(uL + uR)
    return np.where(uL >= uR, np.where(s > 0, fB(uL), fB(uR)),
                    np.where(uL > 0, fB(uL), np.where(uR < 0, fB(uR), 0.0)))

def minmod(a, b, c):
    s = np.sign(a)
    return np.where((s == np.sign(b)) & (s == np.sign(c)),
                    s*np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)

def dgB(N, T=0.4, cfl=0.2):
    """u = +1 left, -1 right: characteristics converge, the shock stands still."""
    h = 1.0/N; xc = (np.arange(N) + 0.5)*h
    u0 = np.where(xc < 0.5, 1.0, -1.0); u1 = np.zeros(N)
    def rhs(u0, u1):
        F = godunov(np.r_[1.0, u0 + u1], np.r_[u0 - u1, -1.0])
        vol = np.zeros(N)
        for q, w in zip(*GL3): vol += fB(u0 + u1*q)*w
        return -(F[1:] - F[:-1])/h, 3.0*(vol - F[1:] - F[:-1])/h
    lim = lambda a, b: minmod(b, np.r_[a[1:], -1.0] - a, a - np.r_[1.0, a[:-1]])
    t = 0.0
    while t < T:
        dt = min(cfl*h/max(np.max(np.abs(u0) + np.abs(u1)), 1e-12), T - t)
        k0, k1 = rhs(u0, u1)
        a0, a1 = u0 + dt*k0, u1 + dt*k1; a1 = lim(a0, a1)
        k0b, k1b = rhs(a0, a1)
        u0 = 0.5*u0 + 0.5*(a0 + dt*k0b); u1 = 0.5*u1 + 0.5*(a1 + dt*k1b)
        u1 = lim(u0, u1); t += dt
    J = (u0 + u1)[:-1] - (u0 - u1)[1:]
    return xc, u0, u1, h, J[N//2 - 1], np.abs(np.delete(J, N//2 - 1)).max()

# ================================================== C: cohesive zone in a bar
Tmax, dfc, Uend = 1.0, 0.1, 0.64

def czm(n):
    h = 0.5/n; nn = 2*(n+1)
    K = np.zeros((nn, nn)); F = np.zeros(nn)
    for e in range(n):
        K[np.ix_([e, e+1], [e, e+1])] += np.array([[1, -1], [-1, 1]])/h
        i = n+1+e
        K[np.ix_([i, i+1], [i, i+1])] += np.array([[1, -1], [-1, 1]])/h
    kc = -Tmax/dfc
    K[np.ix_([n, n+1], [n, n+1])] += np.array([[kc, -kc], [-kc, kc]])
    F[n] += Tmax; F[n+1] -= Tmax
    fixed = {0: 0.0, nn-1: Uend}
    free = [k for k in range(nn) if k not in fixed]
    rhs = F[free] - K[np.ix_(free, list(fixed))] @ np.array(list(fixed.values()))
    U = np.zeros(nn)
    for k, v in fixed.items(): U[k] = v
    U[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
    return (np.linspace(0, .5, n+1), U[:n+1]), (np.linspace(.5, 1, n+1), U[n+1:]), U[n+1]-U[n]

# ======================================================================= plot
fig = plt.figure(figsize=(9.3, 10.6))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1.0], hspace=0.78,
              wspace=0.34, left=0.095, right=0.975, top=0.915, bottom=0.150)
NS = [4, 8, 16, 32, 64]

def dress(ax, xl, yl, ttl, col):
    ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
    ax.set_title(ttl, fontsize=11.5, color=col, pad=6)
    ax.tick_params(labelsize=10); ax.spines[["top", "right"]].set_visible(False)

def verdict(y, txt, col):
    """Verdicts live in the gaps between rows, spanning the full width."""
    fig.text(0.5, y, txt, ha="center", va="center", fontsize=11.5, color=col,
             linespacing=1.45, bbox=dict(fc="white", ec=col, lw=1.3, pad=5.0))

# ---------------------------------------------------------------------- row A
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1])
xx = np.linspace(0, 1, 400)
a1.plot(xx, uexA(xx), ls=(0, (6, 3)), lw=2.0, color=EXACT, label="exact: smooth")
seg, jmp = dgA(4)
for k, ((p, q), (ua, ub)) in enumerate(seg):
    a1.plot([p, q], [ua, ub], color=ELEM, lw=2.6, zorder=3,
            label="DG, 4 elements" if k == 0 else None)
for k, j in enumerate(jmp):                       # EVERY face has a gap
    xf = (k+1)/4
    lo, hi = sorted([seg[k][1][1], seg[k+1][1][0]])
    a1.add_patch(FancyArrowPatch((xf, lo-0.06), (xf, hi+0.06), arrowstyle="<|-|>",
                                 mutation_scale=9, lw=1.4, color=BAD, zorder=6))
    off, ha = [(-8, 10), (-8, 10), (24, -2)][k], ["right", "right", "left"][k]
    a1.annotate(f"{abs(j):.4f}", (xf, (lo+hi)/2), textcoords="offset points",
                xytext=off, ha=ha, va="center", fontsize=10, color=BAD,
                bbox=HALO, zorder=7)
a1.legend(fontsize=10, loc="upper left", frameon=False)
dress(a1, "$x$", "$u(x)$", "A.  DG for $-u''=f$   (second order)", BAD)

for k in range(3):
    g = [abs(dgA(N)[1][int(round((k+1)*N/4))-1]) for N in NS]
    a2.loglog(NS, g, "o-", lw=1.8, ms=5.5, color=plt.cm.Reds(0.45 + 0.2*k),
              label=f"face $x={(k+1)/4:.2f}$")
a2.legend(fontsize=9.5, frameon=False, loc="lower left")
a2.set_xticks(NS); a2.set_xticklabels([str(n) for n in NS])
dress(a2, "elements", "gap", "all three collapse", BAD)
verdict(0.676, "the exact answer has no jump anywhere, so each gap is a deviation "
                "from it\n"
                "$\\Rightarrow$   the gap is ERROR, and refinement destroys it", BAD)

# ---------------------------------------------------------------------- row B
b1 = fig.add_subplot(gs[1, 0]); b2 = fig.add_subplot(gs[1, 1])
xc, u0, u1, h, Jm, Jo = dgB(16)
b1.plot([0, 0.5, 0.5, 1], [1, 1, -1, -1], ls=(0, (6, 3)), lw=2.4, color=EXACT,
        label="exact: a real shock")
for e in range(len(xc)):
    b1.plot([xc[e]-h/2, xc[e]+h/2], [u0[e]-u1[e], u0[e]+u1[e]], color=ELEM, lw=2.6,
            zorder=3, label="DG, 16 elements" if e == 0 else None)
b1.add_patch(FancyArrowPatch((0.5, -1), (0.5, 1), arrowstyle="<|-|>",
                             mutation_scale=11, lw=1.8, color=GOOD, zorder=6))
b1.annotate(f"jump $={Jm:.4f}$", (0.5, 0), textcoords="offset points", xytext=(12, 0),
            ha="left", fontsize=11, color=GOOD, bbox=HALO, zorder=7)
b1.set_ylim(-1.6, 1.6); b1.legend(fontsize=10, loc="lower left", frameon=False)
dress(b1, "$x$", "$u(x)$", "B.  DG for a standing Burgers shock   (first order)", GOOD)

Jms = [dgB(N)[4] for N in NS]
b2.plot(NS, Jms, "o-", lw=2.0, ms=7, color=GOOD)
b2.set_xscale("log"); b2.set_ylim(0, 2.6); b2.set_xticks(NS)
b2.set_xticklabels([str(n) for n in NS])
b2.annotate("exactly $2$ on every mesh\n(every other face: exactly $0$)",
            (NS[1], 2.0), textcoords="offset points", xytext=(6, -40),
            fontsize=10.5, color=GOOD)
dress(b2, "elements", "jump at $x=1/2$", "it does not budge", GOOD)
verdict(0.377, "the exact answer really does have this shock, and the scheme "
                "reproduces it\n"
                "$\\Rightarrow$   the jump is PHYSICS, and refinement cannot touch it", GOOD)

# ---------------------------------------------------------------------- row C
c1 = fig.add_subplot(gs[2, 0]); c2 = fig.add_subplot(gs[2, 1])
(xl, ul), (xr, ur), d = czm(4)
c1.plot([0, 0.5], [0, 0.3], ls=(0, (6, 3)), lw=2.0, color=EXACT,
        label="exact: jumps at the crack")
c1.plot([0.5, 1], [0.3+d, Uend], ls=(0, (6, 3)), lw=2.0, color=EXACT)
c1.plot(xl, ul, color=ELEM, lw=2.6, zorder=3, label="FE, 4+4 elements")
c1.plot(xr, ur, color=ELEM, lw=2.6, zorder=3)
c1.add_patch(FancyArrowPatch((0.5, ul[-1]), (0.5, ur[0]), arrowstyle="<|-|>",
                             mutation_scale=11, lw=1.8, color=GOOD, zorder=6))
c1.annotate(f"gap $={d:.4f}$", (0.5, (ul[-1]+ur[0])/2), textcoords="offset points",
            xytext=(14, -8), ha="left", fontsize=11, color=GOOD, bbox=HALO, zorder=7)
c1.legend(fontsize=10, loc="upper left", frameon=False)
dress(c1, "$x$", "$u(x)$", "C.  cohesive zone in a bar   (interface law)", GOOD)

ds = [czm(N)[2] for N in NS]
c2.plot(NS, ds, "o-", lw=2.0, ms=7, color=GOOD)
c2.set_xscale("log"); c2.set_ylim(0, 0.085); c2.set_xticks(NS)
c2.set_xticklabels([str(n) for n in NS])
c2.annotate("$0.0400$ on every mesh", (NS[1], ds[1]), textcoords="offset points",
            xytext=(-4, 20), fontsize=10.5, color=GOOD)
dress(c2, "elements", "crack opening", "it does not budge either", GOOD)
verdict(0.064, "the jump is a modelled unknown, with its own law $T(\\delta)$ and "
                "its own $G_c$\n"
                "$\\Rightarrow$   the gap is PHYSICS --- it is the crack opening", GOOD)

fig.suptitle("The question is never whether the field jumps. It is whether the "
             "TRUE answer jumps there.", fontsize=13.5, y=0.973)
fig.savefig("q4_three_jumps.pdf")
print("wrote q4_three_jumps.pdf")
print(f"  A  gaps at the three faces (4 elts): {[f'{abs(v):.4f}' for v in dgA(4)[1]]}")
print(f"  B  shock jump by mesh              : {[f'{v:.10f}' for v in Jms]}")
print(f"  B  largest jump anywhere else      : {max(dgB(N)[5] for N in NS):.3e}")
print(f"  C  crack opening by mesh           : {[f'{v:.4f}' for v in ds]}")
