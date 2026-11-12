"""Polynomial degree and continuity class are two different knobs.

Four panels, two elements each, the SAME nodal values at the shared node.
(a)->(b)->(c) turn the degree knob: 1, 2, 3. The kink at the joint never goes.
(d) has the same degree as (c) and no kink, because the slope was made a
shared nodal unknown instead.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, A, B, KINK, OKC = "#15284a", "#1f6fb4", "#c0622a", "#b02020", "#2e7d32"
HALO = dict(fc="white", ec="none", alpha=0.82, pad=0.3)

def lagrange_vals(xn, yn, xs):
    """evaluate the Lagrange interpolant through (xn,yn) at xs"""
    out = np.zeros_like(xs)
    for i, xi in enumerate(xn):
        t = np.ones_like(xs)
        for j, xj in enumerate(xn):
            if i != j: t = t*(xs-xj)/(xi-xj)
        out += yn[i]*t
    return out

# nodal values are chosen so the two sides genuinely disagree on the slope
DATA = {
    1: (([0.0, 1.0], [0.20, 1.00]), ([1.0, 2.0], [1.00, 1.22])),
    2: (([0.0, 0.5, 1.0], [0.20, 0.86, 1.00]),
        ([1.0, 1.5, 2.0], [1.00, 1.45, 1.22])),
    3: (([0.0, 1/3, 2/3, 1.0], [0.20, 0.72, 0.97, 1.00]),
        ([1.0, 4/3, 5/3, 2.0], [1.00, 1.38, 1.47, 1.22])),
}

def slope_at(xn, yn, x0, h=1e-6):
    return (lagrange_vals(np.array(xn), np.array(yn), np.array([x0+h]))[0]
            - lagrange_vals(np.array(xn), np.array(yn), np.array([x0-h]))[0])/(2*h)

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10.6, 10.9))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.64],
              left=0.065, right=0.975, top=0.828, bottom=0.078,
              wspace=0.16, hspace=0.62)
axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                 [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
axv = fig.add_subplot(gs[2, :])

def frame(ax, title, sub, formula, shared, colour):
    ax.set_xlim(-0.18, 2.18); ax.set_ylim(-0.12, 2.05)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([])
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=13.4, color=INK, fontweight="bold", pad=16)
    ax.text(1.0, 1.90, sub, fontsize=11.6, color=colour, ha="center",
            fontweight="bold")
    ax.text(0.5, -0.145, formula, fontsize=10.6, color="0.3", ha="center",
            va="top", transform=ax.transAxes, clip_on=False)
    ax.text(0.5, -0.255, shared, fontsize=11.0, color=colour, ha="center",
            va="top", fontweight="bold", transform=ax.transAxes,
            clip_on=False)

def draw_pair(ax, p):
    (xa, ya), (xb, yb) = DATA[p]
    sa = np.linspace(0, 1, 200); sb = np.linspace(1, 2, 200)
    ax.plot(sa, lagrange_vals(np.array(xa), np.array(ya), sa), color=A, lw=3.0)
    ax.plot(sb, lagrange_vals(np.array(xb), np.array(yb), sb), color=B, lw=3.0)
    for xx, yy, c in [(xa, ya, A), (xb, yb, B)]:
        ax.plot(xx, yy, "o", ms=8.5, color=c, zorder=6)
    ml = slope_at(xa, ya, 1 - 1e-3); mr = slope_at(xb, yb, 1 + 1e-3)
    for m, c, x0, x1 in [(ml, A, 0.62, 1.0), (mr, B, 1.0, 1.38)]:
        xs = np.array([x0, x1])
        ax.plot(xs, 1.0 + m*(xs-1.0), "--", color=c, lw=1.7, alpha=0.95, zorder=5)
    ax.plot([1, 1], [-0.12, 2.0], ":", color="0.72", lw=1.2, zorder=1)
    ax.annotate("kink", (1.0, 1.0), textcoords="offset points", xytext=(2, -46),
                fontsize=12.0, color=KINK, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=KINK, lw=1.9))
    return ml, mr

# ---- (a) linear ----
ax = axes[0, 0]; ml, mr = draw_pair(ax, 1)
frame(ax, "(a)  LINEAR Lagrange", "degree 1    $C^{0}$",
      "$N=\\frac{1-\\xi}{2},\\ \\frac{1+\\xi}{2}$   (2 nodes)",
      "shared at the node:  the value", KINK)
ax.text(1.0, 1.72, f"slope {ml:.2f}  vs  {mr:.2f}", fontsize=10.8, color=KINK,
        ha="center", bbox=HALO)

# ---- (b) quadratic ----
ax = axes[0, 1]; ml, mr = draw_pair(ax, 2)
frame(ax, "(b)  QUADRATIC Lagrange", "degree 2    still $C^{0}$",
      "$N=\\frac{\\xi(\\xi-1)}{2},\\ 1-\\xi^{2},\\ \\frac{\\xi(\\xi+1)}{2}$   (3 nodes)",
      "shared at the node:  the value", KINK)
ax.text(1.0, 1.72, f"slope {ml:.2f}  vs  {mr:.2f}", fontsize=10.8, color=KINK,
        ha="center", bbox=HALO)

# ---- (c) cubic Lagrange ----
ax = axes[1, 0]; ml, mr = draw_pair(ax, 3)
frame(ax, "(c)  CUBIC Lagrange", "degree 3    still $C^{0}$",
      "4 nodes at $\\xi=-1,-\\frac{1}{3},\\frac{1}{3},1$",
      "shared at the node:  the value", KINK)
ax.text(1.0, 1.72, f"slope {ml:.2f}  vs  {mr:.2f}", fontsize=10.8, color=KINK,
        ha="center", bbox=HALO)

# ---- (d) cubic Hermite ----
ax = axes[1, 1]
(xa, ya), (xb, yb) = DATA[3]
m_shared = 0.55                      # ONE slope, owned by the node
ends = [(0.20, 1.00, 1.15, m_shared, 0.0, 1.0, A),
        (1.00, 1.22, m_shared, -0.35, 1.0, 2.0, B)]
for w0, w1, t0, t1, x0, x1, col in ends:
    L = x1 - x0
    s = np.linspace(0, 1, 200)
    H = (w0*(1-3*s**2+2*s**3) + t0*L*(s-2*s**2+s**3)
         + w1*(3*s**2-2*s**3) + t1*L*(-s**2+s**3))
    ax.plot(x0 + L*s, H, color=col, lw=3.0)
ax.plot([0, 1, 2], [0.20, 1.00, 1.22], "o", ms=8.5, color=INK, zorder=6)
xs = np.array([0.58, 1.42])
ax.plot(xs, 1.0 + m_shared*(xs-1.0), "--", color=OKC, lw=2.1, zorder=5)
ax.plot([1, 1], [-0.12, 2.0], ":", color="0.72", lw=1.2, zorder=1)
ax.annotate("one tangent", (1.0, 1.0), textcoords="offset points",
            xytext=(4, -46), fontsize=12.0, color=OKC, fontweight="bold",
            ha="center", arrowprops=dict(arrowstyle="->", color=OKC, lw=1.9))
frame(ax, "(d)  CUBIC Hermite", "degree 3    but $C^{1}$",
      "$H=1-3s^{2}+2s^{3},\\ L(s-2s^{2}+s^{3}),\\ \\ldots$   (2 nodes)",
      "shared at the node:  the value AND the slope", OKC)
ax.text(1.0, 1.72, "slope  0.55  vs  0.55", fontsize=10.8, color=OKC,
        ha="center", bbox=HALO)

# ================= the verdict strip: the same four, judged by two PDEs ======
axv.set_xlim(0, 100); axv.set_ylim(0, 34); axv.axis("off")
COLS = [(43, "(a) linear\n$C^{0}$"), (59, "(b) quadratic\n$C^{0}$"),
        (75, "(c) cubic\n$C^{0}$"), (91, "(d) Hermite\n$C^{1}$")]
for cx, lab in COLS:
    axv.text(cx, 29.5, lab, fontsize=11.8, color=INK, ha="center", va="center",
             fontweight="bold")
axv.plot([2, 97], [24.2, 24.2], color="0.55", lw=1.3)
axv.plot([2, 97], [2.2, 2.2], color="0.55", lw=1.3)

def tick(ax, x, y, good, s=2.5):
    if good:
        ax.plot([x-s, x-s*0.15, x+s*1.05], [y+0.1, y-s*0.85, y+s*0.95],
                color=OKC, lw=3.4, solid_capstyle="round", zorder=5)
    else:
        for d in (1, -1):
            ax.plot([x-s*0.85, x+s*0.85], [y-d*s*0.85, y+d*s*0.85],
                    color=KINK, lw=3.4, solid_capstyle="round", zorder=5)

ROWS = [(17.0, "Poisson,  order 2", "$\\int k\\,u'v'$  needs  $u\\in H^{1}$  ->  $C^{0}$",
         [True, True, True, True]),
        (7.2, "Euler-Bernoulli beam,  order 4",
         "$\\int EI\\,w''v''$  needs  $w\\in H^{2}$  ->  $C^{1}$",
         [False, False, False, True])]
for ry, name, need, verdicts in ROWS:
    axv.text(2, ry+2.0, name, fontsize=12.6, color=INK, ha="left",
             va="center", fontweight="bold")
    axv.text(2, ry-2.2, need, fontsize=11.0, color="0.35", ha="left", va="center")
    for (cx, _), v in zip(COLS, verdicts):
        tick(axv, cx, ry, v)
fig.text(0.5, 0.028,
         "turning the DEGREE knob, (a) to (c), changes nothing in either row.\n"
         "Only the CONTINUITY knob, (d), does.",
         fontsize=12.4, color=OKC, ha="center", va="bottom",
         fontweight="bold", linespacing=1.45)
fig.text(0.5, 0.972, "degree and continuity are two different knobs",
         fontsize=16.5, color=INK, ha="center", fontweight="bold")
fig.text(0.5, 0.952,
         "(a) to (c) turns the DEGREE knob --- the curve gets wigglier inside\n"
         "each element, and the kink never leaves.",
         fontsize=11.8, color="0.3", ha="center", va="top", linespacing=1.4)
fig.text(0.5, 0.902,
         "(c) and (d) are BOTH degree 3.  Only (d) makes the slope a nodal\n"
         "unknown, so the two elements read the same one.",
         fontsize=11.8, color=OKC, ha="center", va="top", fontweight="bold",
         linespacing=1.4)
fig.savefig("q4_degree_vs_continuity.pdf")
print("wrote q4_degree_vs_continuity.pdf")
for p in (1, 2, 3):
    (xa, ya), (xb, yb) = DATA[p]
    print(f"  degree {p}: slope left {slope_at(xa,ya,1-1e-3):+.3f}   "
          f"right {slope_at(xb,yb,1+1e-3):+.3f}")
