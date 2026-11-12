"""Geometry as a variable: four one-parameter families, det J at nodes and Gauss points."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

g = 1/np.sqrt(3)
GP   = [(-g, -g), (g, -g), (g, g), (-g, g)]
CORN = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

def detJ(X, Y, r, s):
    dNr = np.array([-(1-s),  (1-s), (1+s), -(1+s)])/4.0
    dNs = np.array([-(1-r), -(1+r), (1+r),  (1-r)])/4.0
    return (dNr@X)*(dNs@Y) - (dNs@X)*(dNr@Y)

def f_size(L):  return np.array([0,L,L,0.0]),            np.array([0,0,L,L*1.0])
def f_shear(k): return np.array([0,1,1+k,k*1.0]),        np.array([0,0,1,1.0])
def f_taper(w): return np.array([0,1,.5+w/2,.5-w/2]),    np.array([0,0,1,1.0])
def f_pull(k):  return np.array([0,1,1-k,0.0]),          np.array([0,0,1-k,1.0])

#      title      subtitle                          sym  fn       sweep                       picks                 node-label shape  caption
FAM = [
 ("1.  SIZE",  "scale by $L$",          "L", f_size,  np.linspace(0.6, 2.0, 200), [0.6,1.0,1.5,2.0],  -1,
  r"$\det J=L^{2}/4$: equal at all four"),
 ("2.  SHEAR", "shear by $k$",     "k", f_shear, np.linspace(0.0, 1.2, 200), [0.0,0.4,0.8,1.2],   0,
  r"$\det J=1/4$ for every $k$: flat"),
 ("3.  TAPER", "top edge to width $w$", "w", f_taper, np.linspace(1.0, 0.12, 200),[1.0,0.7,0.4,0.12],  0,
  r"bottom pair flat, top pair falls"),
 ("4.  PULL",  "node 3 inward by $k$",        "k", f_pull,  np.linspace(0.0, 0.9, 200), [0.0,0.3,0.5,0.8],   0,
  r"node 3 hits $\det J=0$ at $k=1/2$"),
]
SH   = plt.cm.viridis(np.linspace(0.05, 0.78, 4))
DASH = [(0, ()), (0, (6, 2)), (0, (1.4, 1.4)), (0, (7, 2, 1.4, 2))]
NC   = ["#1f4e9c", "#b5651d", "#2e7d32", "#7b1fa2"]

fig = plt.figure(figsize=(7.2, 4.85))
gs  = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.92],
               hspace=0.50, wspace=0.34, left=0.075, right=0.985, top=0.885, bottom=0.135)

for c, (name, sub, sym, shape, sweep, picks, ni, cap) in enumerate(FAM):
    # ---------------- top: the shapes ----------------
    a = fig.add_subplot(gs[0, c])
    xs, ys, handles = [], [], []
    for i, p in enumerate(picks):
        X, Y = shape(p)
        a.plot(np.append(X, X[0]), np.append(Y, Y[0]), color=SH[i], lw=1.8, zorder=4-i*0.1)
        xs += list(X); ys += list(Y)
        handles.append(Line2D([], [], color=SH[i], lw=1.8, label=f"${sym}={p:g}$"))
    Xn, Yn = shape(picks[ni])
    for i, lab in enumerate("1234"):
        a.annotate(lab, (Xn[i], Yn[i]), textcoords="offset points",
                   xytext=[(-8.5,-9),(7.5,-9),(7.5,6),(-8.5,6)][i], fontsize=7, color="0.35")
    cx, cy = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
    hw = 0.60*max(max(xs)-min(xs), max(ys)-min(ys)) + 0.16
    a.set_xlim(cx-hw, cx+hw); a.set_ylim(cy-hw, cy+hw); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color("0.86")
    a.set_title(f"{name}\n{sub}", fontsize=8.5, pad=4)
    a.legend(handles=handles, fontsize=6.3, ncol=2, loc="upper center",
             bbox_to_anchor=(0.5, -0.01), frameon=False, handlelength=1.0,
             handletextpad=0.35, columnspacing=0.9, labelspacing=0.2)

    # ---------------- bottom: det J ----------------
    b = fig.add_subplot(gs[1, c])
    for r, s in GP:
        b.plot(sweep, [detJ(*shape(p), r, s) for p in sweep], color="0.74", lw=2.6, zorder=1)
    for n, (r, s) in enumerate(CORN):
        b.plot(sweep, [detJ(*shape(p), r, s) for p in sweep],
               color=NC[n], lw=1.4, ls=DASH[n], zorder=3, label=f"node {n+1}")
    b.axhline(0, color="#c62828", lw=0.9, ls=(0, (5, 3)), zorder=2)
    if c == 2: b.invert_xaxis()
    if c == 3:
        kc, kg = 0.5, (3 - np.sqrt(3))/2            # corner zero, then Gauss zero
        b.axvspan(kc, 0.9, color="#c62828", alpha=0.07, lw=0, zorder=0)
        b.axvspan(kc, kg,  color="#c62828", alpha=0.17, lw=0, zorder=0)
        b.axvline(kc, color="0.25", lw=0.8, ls=":", zorder=2)
        b.annotate("tangled", (0.63, 0.195), fontsize=6.6, color="#c62828")
        b.annotate("dark band: tangled, yet\nall 4 Gauss pts $>0$", (0.015, -0.215),
                   fontsize=5.9, color="#c62828", ha="left", va="bottom")
    b.set_xlabel(f"${sym}$", fontsize=9, labelpad=1)
    if c == 0:
        b.set_ylabel(r"$\det J$", fontsize=9, labelpad=2)
    b.tick_params(labelsize=6.6, pad=1.5)
    b.spines[["top", "right"]].set_visible(False)
    b.set_title(cap, fontsize=6.9, color="0.32", pad=3)

key = [Line2D([], [], color=NC[n], lw=1.4, ls=DASH[n], label=f"node {n+1}") for n in range(4)]
key += [Line2D([], [], color="0.74", lw=2.6, label="the 4 Gauss pts"),
        Line2D([], [], color="#c62828", lw=0.9, ls=(0, (5, 3)), label="$\\det J=0$")]
fig.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=6,
           fontsize=7, frameon=False, handlelength=2.2, handletextpad=0.4,
           columnspacing=1.5)
fig.savefig("q4_geomstudy.pdf"); print("wrote q4_geomstudy.pdf")
