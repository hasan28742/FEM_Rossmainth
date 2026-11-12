"""Three objects, three symbols: u, u_a, u_h.

The figure exists to make one thing unmissable: u and u_h are different kinds
of object from u_a, and they are not each other.

  A  the element in the x-y plane -- where the four numbers u_a come from
  B  a LINEAR field: the interpolant lies exactly on the exact surface
     (this is equation (19) -- nothing to see, which IS the point)
  C  a NON-linear field: the same four numbers, and now u_h misses u
  D  that miss, mapped: where u - u_h lives and how big it gets

Laid out 2x2 rather than 1x3, because the page is 6.6 in wide and a 1x3 strip
scales down until nothing inside it can be read.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from matplotlib.gridspec import GridSpec

EXACT, UH, NODE, MID = "#1f6fb4", "#c0622a", "#15284a", "#2e7d32"
XD = np.array([0.0, 3.0, 4.0, 1.0])                # element D, counter-clockwise
YD = np.array([0.0, 0.0, 4.0, 5.0])
HALO = dict(fc="white", ec="none", alpha=0.80, pad=0.4)

def N(r, s):
    return np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])/4.0

def xy(r, s):
    n = N(r, s); return n @ XD, n @ YD

lin = lambda x, y: 2.0 + 3.0*x - 1.0*y             # the field of equation (19)
qua = lambda x, y: x*x                             # anything curved will do

t  = np.linspace(-1, 1, 61)                        # the parametric grid, mapped
R, S = np.meshgrid(t, t)
NG = np.array([(1-R)*(1-S), (1+R)*(1-S), (1+R)*(1+S), (1-R)*(1+S)])/4.0
XG = np.einsum("ajk,a->jk", NG, XD)
YG = np.einsum("ajk,a->jk", NG, YD)

def surfaces(f):
    """exact field over the element, and the Q4 interpolant of its nodal values"""
    ua = f(XD, YD)
    return f(XG, YG), np.einsum("ajk,a->jk", NG, ua), ua

fig = plt.figure(figsize=(10.4, 9.1))
gs  = GridSpec(2, 2, figure=fig, wspace=0.20, hspace=0.46,
               left=0.065, right=0.925, top=0.895, bottom=0.175)

# ================= A: where the four numbers come from =================
axA = fig.add_subplot(gs[0, 0])
ZA, _, uaA = surfaces(lin)
axA.contour(XG, YG, ZA, levels=11, colors=EXACT, linewidths=1.0, alpha=0.55)
axA.plot(np.append(XD, XD[0]), np.append(YD, YD[0]), "-", color=NODE, lw=2.6, zorder=3)
cen = (XD.mean(), YD.mean())
for a in range(4):
    axA.plot(XD[a], YD[a], "o", ms=17, color=NODE, zorder=8)
    axA.annotate(str(a+1), (XD[a], YD[a]), ha="center", va="center", fontsize=9.5,
                 color="white", fontweight="bold", zorder=9)
    dx, dy = XD[a]-cen[0], YD[a]-cen[1]; n = np.hypot(dx, dy)
    axA.annotate(f"$({XD[a]:g},{YD[a]:g})$\n$u_{{{a+1}}}={uaA[a]:g}$",
                 (XD[a], YD[a]), textcoords="offset points",
                 xytext=(34*dx/n, 30*dy/n), ha="center", va="center",
                 fontsize=11.4, color=NODE, zorder=9, bbox=HALO)
px, py = xy(0, 0)
axA.plot(px, py, "D", ms=11, mfc="white", mec=MID, mew=2.4, zorder=8)
axA.annotate(f"$(0,0)\\mapsto({px:.2f},{py:.2f})$\n$u_h={lin(px,py):.2f}$",
             (px, py), textcoords="offset points", xytext=(30, -30), ha="left",
             va="center", fontsize=11.0, color=MID, zorder=9, bbox=HALO,
             arrowprops=dict(arrowstyle="-", color=MID, lw=1.1, shrinkA=2, shrinkB=7))
axA.annotate("contours of\nthe exact $u$", (XG[40, 6], YG[40, 6]),
             textcoords="offset points", xytext=(-26, 2), ha="right", va="center",
             fontsize=10.8, color=EXACT, zorder=9, bbox=HALO,
             arrowprops=dict(arrowstyle="-", color=EXACT, lw=1.0, alpha=0.85,
                             shrinkA=2, shrinkB=2))
axA.set_title("A.  the four numbers the element is given\n"
              r"$u_a=u(x_a,y_a)$ --- read off the blue contours",
              fontsize=13.0, pad=8)
axA.set_xlabel("$x$", fontsize=14); axA.set_ylabel("$y$", fontsize=14)
axA.set_xlim(-4.3, 8.3); axA.set_ylim(-2.6, 6.9); axA.set_aspect("equal")
axA.tick_params(labelsize=11); axA.spines[["top", "right"]].set_visible(False)

# ================= B and C: the two surfaces =================
def panel(ax, f, title):
    Z, Zh, ua = surfaces(f)
    ax.plot_surface(XG, YG, Z, color=EXACT, alpha=0.42, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot_wireframe(XG, YG, Zh, rstride=7, cstride=7, color=UH, linewidth=1.3)
    ax.scatter(XD, YD, ua, s=74, color=NODE, depthshade=False)
    span = Z.max() - Z.min()
    for a in range(4):
        ax.text(XD[a], YD[a], ua[a] + 0.10*span, f"$u_{{{a+1}}}$", fontsize=12.0,
                color=NODE, fontweight="bold", ha="center")
    ax.set_title(title, fontsize=13.0, pad=-2)
    ax.set_xlabel("$x$", fontsize=12, labelpad=-5)
    ax.set_ylabel("$y$", fontsize=12, labelpad=-5)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=9.5, pad=-2)
    ax.set_box_aspect((1, 1, 0.66))
    return np.abs(Z - Zh).max()

axB = fig.add_subplot(gs[0, 1], projection="3d")
eB  = panel(axB, lin, "B.  a LINEAR field,  $u=2+3x-y$")
axB.text2D(0.5, -0.02, f"the two surfaces coincide:  $u_h=u$\n"
                       f"$\\max|u-u_h|={eB:.1e}$",
           transform=axB.transAxes, ha="center", va="top",
           fontsize=12.0, color="#2e7d32", fontweight="bold")

axC = fig.add_subplot(gs[1, 0], projection="3d")
eC  = panel(axC, qua, "C.  a CURVED field,  $u=x^{2}$")
axC.text2D(0.5, -0.02, f"the flat sheet cannot bend:  $u_h\\neq u$\n"
                       f"$\\max|u-u_h|={eC:.3f}$",
           transform=axC.transAxes, ha="center", va="top",
           fontsize=12.0, color="#b02020", fontweight="bold")

# ================= D: that error, mapped =================
axD = fig.add_subplot(gs[1, 1])
ZC, ZhC, uaC = surfaces(qua)
ERR = ZhC - ZC
cf = axD.contourf(XG, YG, ERR, levels=12, cmap="OrRd")
axD.contour(XG, YG, ERR, levels=12, colors="white", linewidths=0.45, alpha=0.6)
axD.plot(np.append(XD, XD[0]), np.append(YD, YD[0]), "-", color=NODE, lw=2.6, zorder=3)
cb = fig.colorbar(cf, ax=axD, fraction=0.044, pad=0.03)
cb.set_label("$u_h-u$", fontsize=12); cb.ax.tick_params(labelsize=9.5)
for a in range(4):
    axD.plot(XD[a], YD[a], "o", ms=11, color=NODE, zorder=8)
    dx, dy = XD[a]-cen[0], YD[a]-cen[1]; n = np.hypot(dx, dy)
    axD.annotate("$0$", (XD[a], YD[a]), textcoords="offset points",
                 xytext=(19*dx/n, 17*dy/n), ha="center", va="center",
                 fontsize=11.4, color=NODE, zorder=9, bbox=HALO)
i = np.unravel_index(np.abs(ERR).argmax(), ERR.shape)
axD.plot(XG[i], YG[i], "*", ms=17, color="#4a1010", zorder=9)
axD.annotate(f"worst here: ${ERR[i]:.2f}$\n$u_h=6.50$ vs $u=4.00$",
             (XG[i], YG[i]), textcoords="offset points", xytext=(0, -62),
             ha="center", va="center", fontsize=11.0, color="#4a1010",
             zorder=9, bbox=HALO,
             arrowprops=dict(arrowstyle="-", color="#4a1010", lw=1.1,
                             shrinkA=2, shrinkB=8))
axD.set_title("D.  panel C's miss, mapped\n"
              r"zero at the nodes, worst in between",
              fontsize=13.0, pad=8)
axD.set_xlabel("$x$", fontsize=14); axD.set_ylabel("$y$", fontsize=14)
axD.set_xlim(-1.4, 5.4); axD.set_ylim(-2.4, 6.0); axD.set_aspect("equal")
axD.tick_params(labelsize=11); axD.spines[["top", "right"]].set_visible(False)

fig.text(0.5, 0.962,
         "Same element, same four nodes, same rule $u_h=\\sum_a N_a u_a$ --- "
         "only the sampled field changes.",
         ha="center", fontsize=13.0, color="0.22")
fig.text(0.5, 0.015,
         "$u$: a function, defined everywhere --- what you are after.      "
         "$u_a$: four numbers, one per node.\n"
         "$u_h=\\sum_a N_a u_a$: the function the element rebuilds out of them.  "
         "In B it equals $u$; in C and D it does not.",
         ha="center", va="bottom", fontsize=11.4, color="0.25", linespacing=1.7)
fig.savefig("q4_three_us.pdf")
print(f"wrote q4_three_us.pdf   max|u-u_h|: linear {eB:.3e}   quadratic {eC:.6f}")
print(f"worst point: (x,y)=({XG[i]:.4f},{YG[i]:.4f})  u_h={ZhC[i]:.4f}  u={ZC[i]:.4f}")
