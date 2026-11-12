"""Element D: the reference square and its image, with every pairing made explicit.

What the figure has to say without being read:
  * which parent corner is which physical corner, and which Gauss point is which
    -> the numeral is drawn INSIDE the marker, identically in both panels;
  * the element's four edges ARE grid lines -> each carries its own r=const /
    s=const label and its family's colour, and the weights only step down;
  * the two middle members ARE the bimedians -> drawn once, not twice.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

RFAM, SFAM = "#1f6fb4", "#c0622a"        # the r=const and s=const families
NODE, GAUSS, MID = "#15284a", "#55585c", "#2e7d32"
XD = np.array([0.0, 3.0, 4.0, 1.0])      # local nodes 1..4, counter-clockwise
YD = np.array([0.0, 0.0, 4.0, 5.0])
HALO = dict(fc="white", ec="none", alpha=0.75, pad=0.4)

def N(r, s):
    return np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])/4.0

def xy(r, s):
    n = N(r, s); return n @ XD, n @ YD

def detJ(r, s):
    return 3.5 - 3.0*r/8.0 + s/8.0

g  = 1/np.sqrt(3)
GP = [(-g, -g), (g, -g), (g, g), (-g, g)]
GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]       # -1 and +1 are the edges themselves

def badge(ax, P, marker, size, colour, offs=None, coords=None, fs=8.5):
    """Marker with its index drawn inside it -- the cross-panel identity cue."""
    for i, (px, py) in enumerate(P):
        ax.plot(px, py, marker, ms=size, color=colour, zorder=8)
        ax.annotate(str(i+1), (px, py), ha="center", va="center", fontsize=fs,
                    color="white", fontweight="bold", zorder=9)
        if coords is not None:
            ax.annotate(coords[i], (px, py), textcoords="offset points",
                        xytext=offs[i], ha="center", va="center", fontsize=10.6,
                        color=colour, zorder=9, bbox=HALO)

def edge_label(ax, pa, pb, text, out, color, pad=22, fs=8.5, f=0.24):
    """Label an edge along its own direction at fraction f, pushed away from `out`."""
    mx, my = pa[0]+f*(pb[0]-pa[0]), pa[1]+f*(pb[1]-pa[1])
    ang = np.degrees(np.arctan2(pb[1]-pa[1], pb[0]-pa[0]))
    if ang > 90:   ang -= 180
    if ang <= -90: ang += 180
    dx, dy = mx-out[0], my-out[1]; n = np.hypot(dx, dy)
    ax.annotate(text, (mx, my), textcoords="offset points",
                xytext=(pad*dx/n, pad*dy/n), ha="center", va="center",
                rotation=ang, rotation_mode="anchor", fontsize=fs,
                color=color, fontweight="bold", zorder=7, bbox=HALO)

def gridstyle(c):
    """Interior members differ from the edges only in weight, never in kind."""
    if abs(c) == 1.0: return dict(lw=2.8, alpha=1.00, zorder=2)   # an edge
    if c == 0.0:      return dict(lw=2.1, alpha=0.95, zorder=2)   # a bimedian
    return dict(lw=1.5, alpha=0.80, zorder=1)

fig = plt.figure(figsize=(10.6, 5.7))
gs  = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.075, right=0.975,
               top=0.865, bottom=0.285)
axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# ================= left: the reference square =================
for c in GRID:
    axL.plot([c, c], [-1, 1], color=RFAM, **gridstyle(c))
    axL.plot([-1, 1], [c, c], color=SFAM, **gridstyle(c))
edge_label(axL, (-1,-1), (-1,1), "$r=-1$", (0,0), RFAM)
edge_label(axL, ( 1,-1), ( 1,1), "$r=+1$", (0,0), RFAM)
edge_label(axL, (-1,-1), ( 1,-1), "$s=-1$", (0,0), SFAM)
edge_label(axL, (-1, 1), ( 1, 1), "$s=+1$", (0,0), SFAM)
axL.annotate("$r=+0.5$", (0.5, -1), textcoords="offset points", xytext=(0, -20),
             ha="center", va="center", fontsize=9.9, color=RFAM)
axL.annotate("$s=+0.5$", (-1, 0.5), textcoords="offset points", xytext=(-40, 0),
             ha="center", va="center", fontsize=9.9, color=SFAM)
for (mr, ms_), off in [((0,-1), (0,14)), ((1,0), (-28,0)), ((0,1), (0,-14)), ((-1,0), (30,0))]:
    axL.plot(mr, ms_, "o", ms=8, mfc="white", mec=MID, mew=2.0, zorder=7)
    axL.annotate(f"$({mr:g},{ms_:g})$", (mr, ms_), textcoords="offset points", xytext=off,
                 ha="center", va="center", fontsize=9.9, color=MID, zorder=8, bbox=HALO)
axL.plot(0, 0, "D", ms=10, mfc="white", mec=MID, mew=2.3, zorder=7)
axL.annotate("$(0,0)$", (0,0), textcoords="offset points", xytext=(0, 18),
             ha="center", fontsize=10.6, color=MID, zorder=8, bbox=HALO)
badge(axL, GP, "s", 13, GAUSS, fs=7.5)
badge(axL, [(-1,-1), (1,-1), (1,1), (-1,1)], "o", 15, NODE,
      offs=[(-27,-14), (27,-14), (27,14), (-27,14)],
      coords=["$(-1,-1)$", "$(1,-1)$", "$(1,1)$", "$(-1,1)$"])
axL.set_title("reference square: a uniform grid in $(r,s)$\n"
              r"squares are the $2\times2$ Gauss points, $r,s=\pm1/\sqrt{3}$, $w_q=1$",
              fontsize=13.2, pad=7)
axL.set_xlabel("$r$", fontsize=14.5); axL.set_ylabel("$s$", fontsize=14.5)
axL.set_xlim(-1.6, 1.6); axL.set_ylim(-1.6, 1.6); axL.set_aspect("equal")
axL.set_xticks([-1, 0, 1]); axL.set_yticks([-1, 0, 1])
axL.tick_params(labelsize=11.5)
axL.spines[["top","right"]].set_visible(False)

# ================= right: the physical element =================
u = [-1.0, 1.0]                                      # images stay straight
for c in GRID:
    axR.plot(*zip(*[xy(c, t) for t in u]), color=RFAM, **gridstyle(c))
    axR.plot(*zip(*[xy(t, c) for t in u]), color=SFAM, **gridstyle(c))
P   = list(zip(XD, YD))
cen = (XD.mean(), YD.mean())
for (ia, ib), col, lab in [((0,1), SFAM, "$s=-1$"), ((1,2), RFAM, "$r=+1$"),
                           ((3,2), SFAM, "$s=+1$"), ((0,3), RFAM, "$r=-1$")]:
    edge_label(axR, P[ia], P[ib], lab, cen, col)
    L  = np.hypot(XD[ib]-XD[ia], YD[ib]-YD[ia])
    mx = XD[ia]+0.74*(XD[ib]-XD[ia]); my = YD[ia]+0.74*(YD[ib]-YD[ia])
    dx, dy = mx-cen[0], my-cen[1]; n = np.hypot(dx, dy)
    axR.annotate(f"$|{ia+1}{ib+1}|={L:.3f}$", (mx, my), textcoords="offset points",
                 xytext=(24*dx/n, 24*dy/n), ha="center", va="center",
                 fontsize=9.9, color="0.35", zorder=7, bbox=HALO)
for mr, ms_ in [(0,-1), (1,0), (0,1), (-1,0)]:
    mxp, myp = xy(mr, ms_)
    dx, dy = mxp-cen[0], myp-cen[1]; n = np.hypot(dx, dy)
    axR.plot(mxp, myp, "o", ms=8, mfc="white", mec=MID, mew=2.0, zorder=7)
    axR.annotate(f"$({mr:g},{ms_:g})$", (mxp, myp), textcoords="offset points",
                 xytext=(-18*dx/n, -18*dy/n), ha="center", va="center",
                 fontsize=9.9, color=MID, zorder=8, bbox=HALO)
axR.plot(*xy(0,0), "D", ms=10, mfc="white", mec=MID, mew=2.3, zorder=7)
axR.annotate("$(r,s)=(0,0)$", xy(0,0), textcoords="offset points", xytext=(-26, -22),
             ha="center", fontsize=9.9, color=MID, zorder=8, bbox=HALO)
GPx = [xy(r, s) for r, s in GP]
badge(axR, GPx, "s", 13, GAUSS, fs=7.5)
DJOFF = [(0, -19), (-24, -10), (0, -19), (0, -19)]    # q2 nudged clear of node 2
for q, (px, py) in enumerate(GPx):
    axR.annotate(f"{detJ(*GP[q]):.3f}", (px, py), textcoords="offset points",
                 xytext=DJOFF[q], ha="center", va="top", fontsize=9.9,
                 color=GAUSS, fontweight="bold", zorder=8, bbox=HALO)
badge(axR, P, "o", 15, NODE, offs=[(-31,-12), (27,-14), (31,12), (-31,12)],
      coords=["$(0,0)$", "$(3,0)$", "$(4,4)$", "$(1,5)$"])
axR.set_title("element $D$: the image of that grid\n"
              r"number under each Gauss point is $\det J$ there", fontsize=13.2, pad=7)
axR.set_xlabel("$x$", fontsize=14.5); axR.set_ylabel("$y$", fontsize=14.5)
axR.set_xlim(-1.65, 5.75); axR.set_ylim(-1.35, 6.45); axR.set_aspect("equal")
axR.tick_params(labelsize=11.5)
axR.spines[["top","right"]].set_visible(False)

# ================= the map, between the panels =================
fig.add_artist(FancyArrowPatch((0.455, 0.56), (0.545, 0.56), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=23, lw=1.6, color="0.25"))
fig.text(0.50, 0.605, r"$(x,y)=\sum_a N_a\,(x_a,y_a)$", ha="center", fontsize=12.5,
         color="0.25")
fig.text(0.50, 0.515, "node $a\\mapsto$ node $a$\n$q_a\\mapsto q_a$",
         ha="center", va="top", fontsize=11.9, color="0.25")
fig.text(0.5, 0.012,
         "Node $a$ sits at parent corner $a$, and $q_a$ in node $a$'s corner --- that "
         "pairing is the element's numbering, nothing more.\n"
         "The four edges are not separate from the grid --- they are its outermost "
         r"members $r=\pm1$, $s=\pm1$," "\n"
         "and the middle two are the bimedians, crossing at the image of $(0,0)$.\n"
         "Every image line is straight and still evenly divided --- equal steps in $r,s$ "
         "stay equal along each line (bottom $0.750$, top $0.791$).\n"
         "The distortion is the $\\mathit{fan}$ between opposite edges, and the cell areas "
         "it leaves behind: $0.781$ to $0.969$, summing to $14$.",
         ha="center", va="bottom", fontsize=10, color="0.28", linespacing=1.45)
fig.savefig("q4_elemD.pdf")
print("wrote q4_elemD.pdf")
