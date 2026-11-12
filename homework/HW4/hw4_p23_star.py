"""
HW4 Problems 2 and 3 -- star-shaped domain, PyDistMesh + the Canvas CG(1) code.

    Omega = { phi < 0 },   phi(r,theta) = r - 0.75 - 0.25 sin(5 theta)
    -u_xx - u_yy = 1  in Omega,   u = 0  on dOmega

Problem 2: mesh Omega with distmesh2d (uniform size, h0 = 0.1, 0.05, 0.025)
           and check each mesh: area against the exact area, the regularity
           ratio max h_j/rho_j (Lecture 13), and that the nodes bnd2d marks as
           boundary (|phi| <= 1e-8) are exactly the nodes on boundary edges.
           For this phi they are not: distmesh2d's projection stops once
           phi <= 1e-14 and leaves valley-side boundary nodes (r <= 0.78)
           up to 1e-4 inside.  Fix (the bundle is not edited): before bnd2d, every node
           on a boundary edge is moved radially onto r = 0.75 + 0.25 sin(5 th).
Problem 3: solve with assemble.stiffness_matrix / assemble.rhs, unchanged.
           Before that, the same pipeline on the unit disc, where the exact
           solution (1 - x^2 - y^2)/4 is known, as a check of the code.

Writes hw4_p2_mesh.pdf, hw4_p2_table.tex, hw4_p3_solution.pdf,
       hw4_p3_table.tex, hw4_p3_disc_table.tex.
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

np.random.seed(0)
BLUE, BROWN = "#1f4e9c", "#b5651d"


# ------------------------------------------------------------ geometry
def fd_star(p):
    """phi(r,theta) = r - 0.75 - 0.25 sin(5 theta), written in x, y."""
    r = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    th = np.arctan2(p[:, 1], p[:, 0])
    return r - 0.75 - 0.25*np.sin(5.0*th)


def fd_disc(p):
    return np.sqrt((p**2).sum(1)) - 1.0


AREA_STAR = 0.5*(2*np.pi*0.75**2 + np.pi*0.25**2)   # (1/2) int r(theta)^2 dtheta
AREA_DISC = np.pi


# ------------------------------------------------------------ mesh checks
def regularity(p, t, area):
    a = np.linalg.norm(p[t[:, 1]] - p[t[:, 0]], axis=1)
    b = np.linalg.norm(p[t[:, 2]] - p[t[:, 1]], axis=1)
    c = np.linalg.norm(p[t[:, 0]] - p[t[:, 2]], axis=1)
    hj = np.max(np.vstack((a, b, c)), axis=0)
    rhoj = 2.0*area/(0.5*(a + b + c))           # inscribed-circle diameter
    return hj/rhoj


def boundary_edge_nodes(t):
    """nodes lying on an edge that belongs to exactly one triangle"""
    e = np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return set(np.unique(uniq[cnt == 1]).tolist())


def snap_star(p, t):
    """move every node on a boundary edge radially onto phi = 0"""
    p = p.copy()
    k = np.array(sorted(boundary_edge_nodes(t)))
    th = np.arctan2(p[k, 1], p[k, 0])
    R = 0.75 + 0.25*np.sin(5.0*th)
    p[k, 0], p[k, 1] = R*np.cos(th), R*np.sin(th)
    return p


def mesh(fd, h0, snap=None, raw=None):
    t0 = time.time()
    if raw is None:
        p, t = dm.distmesh2d(fd, distfunc.huniform, h0, (-1, -1, 1, 1))
    else:
        p, t = raw
    praw, traw = p.copy(), t.copy()
    if snap is not None:
        p = snap(p, t)
    p, t, nin, nbd, area = bnd.bnd2d(fd, p, t, -1.0, -1.0, 1.0, 1.0)
    tm = time.time() - t0
    # the two checks the solver relies on
    bnodes_topo = boundary_edge_nodes(t)
    bnodes_phi = set(range(nin, nin + nbd))
    used = np.zeros(p.shape[0], bool); used[t.ravel()] = True
    ok_bnd = (bnodes_topo == bnodes_phi)
    ok_used = used.all()
    return dict(p=p, t=t, nin=nin, nbd=nbd, area=area, h0=h0, tmesh=tm,
                ok_bnd=ok_bnd, ok_used=ok_used, reg=regularity(p, t, area),
                nbtopo=len(bnodes_topo), raw=(praw, traw))


def solve(m, f=lambda x, y: 1.0):
    A = assemble.stiffness_matrix(m["p"], m["t"], m["nin"], m["area"])
    b = assemble.rhs(m["p"], m["t"], m["nin"], m["area"], f)
    U = np.zeros(m["nin"] + m["nbd"])
    U[:m["nin"]] = scipy.sparse.linalg.spsolve(A, b)
    asym = abs(A - A.T).max()
    return U, asym


def integral(m, U):
    """int_Omega U_h = sum_K |K| * (mean of the three vertex values)"""
    return (m["area"]*U[m["t"]].mean(1)).sum()


H0 = [0.1, 0.05, 0.025]

# ============================================================ disc check
disc_rows = []
for h0 in H0:
    m = mesh(fd_disc, h0)
    U, asym = solve(m)
    p = m["p"]
    uex = 0.25*(1.0 - p[:, 0]**2 - p[:, 1]**2)
    err = np.abs(U - uex).max()
    disc_rows.append((h0, m["nin"], m["nbd"], m["t"].shape[0], U.max(), err))
    print("disc h0=%.3f int=%5d bnd=%4d el=%5d  maxU=%.6f  max|U-u|=%.3e  "
          "bnd_ok=%s used_ok=%s asym=%.1e  (%.0fs)"
          % (h0, m["nin"], m["nbd"], m["t"].shape[0], U.max(), err,
             m["ok_bnd"], m["ok_used"], asym, m["tmesh"]))

# ============================================================ star
star = []
for h0 in H0:
    m0 = mesh(fd_star, h0)                       # bundle as is
    U0, _ = solve(m0)
    m = mesh(fd_star, h0, snap=snap_star, raw=m0["raw"])   # same nodes, snapped
    U, asym = solve(m)
    m["U"] = U
    m["asym"] = asym
    m["raw_nbd"], m["raw_ok"], m["raw_maxU"] = m0["nbd"], m0["ok_bnd"], U0.max()
    m["raw_mesh"] = m0
    print("   raw: bnd2d flags %d of %d boundary-edge nodes, max U = %.6f"
          % (m0["nbd"], m0["nbtopo"], U0.max()))
    star.append(m)
    k = U.argmax()
    print("star h0=%.3f int=%5d bnd=%4d el=%5d area=%.5f (exact %.5f, rel %+.1e) "
          "max h/rho=%.2f  maxU=%.6f at (%.3f,%.3f)  intU=%.6f  "
          "bnd_ok=%s used_ok=%s asym=%.1e  (%.0fs)"
          % (h0, m["nin"], m["nbd"], m["t"].shape[0], m["area"].sum(),
             AREA_STAR, (m["area"].sum() - AREA_STAR)/AREA_STAR,
             m["reg"].max(), U.max(), m["p"][k, 0], m["p"][k, 1],
             integral(m, U), m["ok_bnd"], m["ok_used"], asym, m["tmesh"]))

# ------------------------------------------------------------ tables
with open("hw4_p2_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}rrrrrrrc@{}}\n\\toprule\n")
    f.write("$h_0$ & interior & boundary & elements & mesh area & "
            "rel.\\ error & $\\max_j h_j/\\rho_j$ & "
            "flagged: raw $\\to$ snapped\\\\\n"
            "\\midrule\n")
    for m in star:
        a = m["area"].sum()
        mant, ex = ("%.1e" % ((a - AREA_STAR)/AREA_STAR)).split("e")
        assert m["ok_bnd"] and m["ok_used"]
        f.write("%.3f & %d & %d & %d & %.5f & $%s\\!\\times\\!10^{%d}$ & %.2f & "
                "%d/%d $\\to$ %d/%d\\\\\n"
                % (m["h0"], m["nin"], m["nbd"], m["t"].shape[0], a,
                   mant, int(ex), m["reg"].max(),
                   m["raw_nbd"], m["nbtopo"], m["nbd"], m["nbtopo"]))
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("hw4_p3_disc_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}rrrrc@{}}\n\\toprule\n")
    f.write("$h_0$ & interior nodes & $\\max U$ & "
            "$\\max_i\\abs{U_i-u(x_i,y_i)}$ & rate\\\\\n\\midrule\n")
    prev = None
    for (h0, nin, nbd, ne, umax, err) in disc_rows:
        rate = "---" if prev is None else "%.2f" % np.log2(prev/err)
        mant, ex = ("%.2e" % err).split("e")
        f.write("%.3f & %d & %.5f & $%s\\!\\times\\!10^{%d}$ & %s\\\\\n"
                % (h0, nin, umax, mant, int(ex), rate))
        prev = err
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("hw4_p3_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}rrrcr@{}}\n\\toprule\n")
    f.write("$h_0$ & unknowns & $\\max U$ & at $(x,y)$ & "
            "$\\max U$, raw mesh\\\\\n\\midrule\n")
    for m in star:
        U = m["U"]; k = U.argmax()
        z = lambda v: round(v, 3) + 0.0          # no "-0.000"
        f.write("%.3f & %d & %.5f & $(%.3f,\\,%.3f)$ & %.5f\\\\\n"
                % (m["h0"], m["nin"], U.max(), z(m["p"][k, 0]), z(m["p"][k, 1]),
                   m["raw_maxU"]))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------ figures
th = np.linspace(0, 2*np.pi, 721)
rr = 0.75 + 0.25*np.sin(5*th)

# Problem 2: the h0 = 0.05 mesh, and a zoom on one concave valley
m = star[1]
p, t, nin = m["p"], m["t"], m["nin"]
fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4),
                       gridspec_kw=dict(width_ratios=[1.25, 1]))
for a in ax:
    a.triplot(p[:, 0], p[:, 1], t, color="k", lw=0.35)
    a.plot(rr*np.cos(th), rr*np.sin(th), color=BROWN, lw=0.8, alpha=0.6)
    a.plot(p[:nin, 0], p[:nin, 1], ".", color=BLUE, ms=2.5)
    a.plot(p[nin:, 0], p[nin:, 1], ".", color=BROWN, ms=4)
    a.set_aspect("equal"); a.tick_params(labelsize=8)
ax[0].set_xlim(-1.05, 1.05); ax[0].set_ylim(-1.05, 1.05)
ax[0].set_title("$h_0=0.05$: %d interior, %d boundary nodes, %d triangles"
                % (nin, m["nbd"], t.shape[0]), fontsize=10)
# valley: sin(5 theta) = -1  ->  theta = 3pi/10 (+ 2pi k/5); r = 0.5
tv = 3*np.pi/10 + 2*np.pi/5
cx, cy = 0.5*np.cos(tv), 0.5*np.sin(tv)
ax[0].add_patch(plt.Rectangle((cx - 0.22, cy - 0.22), 0.44, 0.44,
                              fill=False, ec="r", lw=1))
m0 = m["raw_mesh"]
missed = sorted(set(range(m0["nin"])) & boundary_edge_nodes(m0["t"]))
ax[1].plot(m0["p"][missed, 0], m0["p"][missed, 1], "o", mfc="none",
           mec="r", ms=7, mew=1.2)
ax[0].plot(m0["p"][missed, 0], m0["p"][missed, 1], "o", mfc="none",
           mec="r", ms=4, mew=0.8)
ax[1].set_xlim(cx - 0.22, cx + 0.22); ax[1].set_ylim(cy - 0.22, cy + 0.22)
ax[1].set_title("zoom on a valley; red: missed by bnd2d before the snap",
                fontsize=10)
fig.tight_layout()
fig.savefig("hw4_p2_mesh.pdf")
plt.close(fig)

# Problem 3: the solution on the finest mesh
m = star[2]
p, t, U = m["p"], m["t"], m["U"]
fig, ax = plt.subplots(figsize=(5.6, 4.6))
pc = ax.tripcolor(p[:, 0], p[:, 1], t, U, shading="gouraud", cmap="inferno")
ax.tricontour(p[:, 0], p[:, 1], t, U, colors="k", linewidths=0.5, levels=10)
k = U.argmax()
xk, yk = (0.0 if abs(v) < 5e-4 else v for v in p[k])     # no "-0.000"
ax.plot(xk, yk, "+", color="c", ms=10, mew=1.5, label="maximum $U$")
ax.plot(rr*np.cos(th), rr*np.sin(th), color="k", lw=0.8)
cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("$U$")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title("CG(1), $h_0=%.3f$\nmaximum $U$ value $=%.5f$ at $(x,y)=(%.3f,\\,%.3f)$"
             % (m["h0"], U.max(), xk, yk), fontsize=10)
fig.tight_layout()
fig.savefig("hw4_p3_solution.pdf")
plt.close(fig)
print("wrote hw4_p2_mesh.pdf hw4_p2_table.tex hw4_p3_solution.pdf "
      "hw4_p3_table.tex hw4_p3_disc_table.tex")
