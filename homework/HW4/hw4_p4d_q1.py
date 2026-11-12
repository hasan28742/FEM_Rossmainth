"""
HW4 Problem 4(d) -- bilinear square elements (Problem 4(c)) on [-1,1]^2

    -Laplace(u) = f  in Omega = (-1,1)^2,     u = g = 1  on dOmega
    exact:  u = exp(sin(pi x) sin(pi y))

Uniform N x N grid, h = 2/N, nodes (x_i, y_j) = (-1 + i h, -1 + j h).
Element matrix (Problem 4(c)), corners counter-clockwise from bottom-left:
    k^e = (1/6) [[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]]
Boundary nodes carry the known value g = 1 and are moved to the right-hand
side (Lecture 5); only the (N-1)^2 interior values are solved for.

Load vector, two ways:
    'trap'  : b_ij = h^2 f(x_i,y_j)          (the trapezoidal rule of 4(c))
    'gauss' : b_ij = int f phi_ij, 3x3 Gauss per element
Errors by 4x4 Gauss per element:
    ||e||_L2^2 = int (u-U)^2 ,   ||e||_V^2 = ||e||_L2^2 + int |grad(u-U)|^2

Writes hw4_p4d_table.tex and hw4_p4d_figure.pdf.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PI = np.pi


# ---------------------------------------------------------------- data
def u_exact(x, y):
    return np.exp(np.sin(PI*x)*np.sin(PI*y))


def grad_u(x, y):
    u = u_exact(x, y)
    return (u*PI*np.cos(PI*x)*np.sin(PI*y), u*PI*np.sin(PI*x)*np.cos(PI*y))


def f(x, y):
    sx, sy, cx, cy = np.sin(PI*x), np.sin(PI*y), np.cos(PI*x), np.cos(PI*y)
    return PI**2*np.exp(sx*sy)*(2*sx*sy + 2*cx**2*cy**2 - cx**2 - cy**2)


def g(x, y):
    return np.ones_like(x)


# ------------------------------------------------- the element (4(c))
KE = np.array([[4, -1, -2, -1],
               [-1, 4, -1, -2],
               [-2, -1, 4, -1],
               [-1, -2, -1, 4]], float)/6.0
XI = np.array([0, 1, 1, 0]); ETA = np.array([0, 0, 1, 1])   # corner coords


def shape(xi, eta):
    """N_a(xi,eta) = (1-|xi-xi_a|)(1-|eta-eta_a|) on [0,1]^2, and d/dxi, d/deta"""
    lx = np.where(XI[:, None] == 1, xi[None, :], 1 - xi[None, :])
    ly = np.where(ETA[:, None] == 1, eta[None, :], 1 - eta[None, :])
    dlx = np.where(XI[:, None] == 1, 1.0, -1.0)*np.ones_like(lx)
    dly = np.where(ETA[:, None] == 1, 1.0, -1.0)*np.ones_like(ly)
    return lx*ly, dlx*ly, lx*dly


def gauss2d(n):
    gp, gw = np.polynomial.legendre.leggauss(n)
    gp = 0.5*(gp + 1); gw = 0.5*gw                      # map to [0,1]
    xi, eta = np.meshgrid(gp, gp, indexing="ij")
    w = np.outer(gw, gw)
    return xi.ravel(), eta.ravel(), w.ravel()


def solve(N, load="gauss"):
    h = 2.0/N
    n1 = N + 1
    xs = -1.0 + h*np.arange(n1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")           # X[i,j] = x_i
    node = lambda i, j: i + j*n1

    # element connectivity, counter-clockwise from bottom-left
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    I = I.ravel(); J = J.ravel()
    conn = np.stack([node(I, J), node(I + 1, J), node(I + 1, J + 1),
                     node(I, J + 1)], axis=1)            # (N^2, 4)

    # stiffness: same k^e for every element (h cancels in 2D)
    rows = np.repeat(conn, 4, axis=1).ravel()
    cols = np.tile(conn, (1, 4)).ravel()
    vals = np.tile(KE.ravel(), N*N)
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n1*n1, n1*n1))

    # load
    if load == "trap":
        b = h*h*f(X, Y).ravel(order="F")
    else:
        xi, eta, w = gauss2d(3)
        Nv, _, _ = shape(xi, eta)                        # (4, nq)
        x0 = X.ravel(order="F")[conn[:, 0]]; y0 = Y.ravel(order="F")[conn[:, 0]]
        fq = f(x0[:, None] + h*xi[None, :], y0[:, None] + h*eta[None, :])
        be = h*h*(fq*w[None, :]) @ Nv.T                  # (N^2, 4)
        b = np.bincount(conn.ravel(), weights=be.ravel(), minlength=n1*n1)

    # interior / boundary split; boundary values moved to the RHS
    onb = np.zeros((n1, n1), bool)
    onb[0, :] = onb[-1, :] = onb[:, 0] = onb[:, -1] = True
    onb = onb.ravel(order="F")
    inn = ~onb
    gB = g(X, Y).ravel(order="F")[onb]
    KII = K[inn][:, inn]
    rhs = b[inn] - K[inn][:, onb] @ gB
    U = np.empty(n1*n1)
    U[onb] = gB
    U[inn] = spla.spsolve(KII.tocsc(), rhs)

    # errors
    xi, eta, w = gauss2d(4)
    Nv, Nxi, Neta = shape(xi, eta)
    Ue = U[conn]                                         # (N^2, 4)
    x0 = X.ravel(order="F")[conn[:, 0]]; y0 = Y.ravel(order="F")[conn[:, 0]]
    xq = x0[:, None] + h*xi[None, :]; yq = y0[:, None] + h*eta[None, :]
    Uq = Ue @ Nv
    Uxq = Ue @ Nxi/h; Uyq = Ue @ Neta/h
    ux, uy = grad_u(xq, yq)
    eL2 = np.sqrt(h*h*np.sum(w[None, :]*(u_exact(xq, yq) - Uq)**2))
    eH1s = np.sqrt(h*h*np.sum(w[None, :]*((ux - Uxq)**2 + (uy - Uyq)**2)))
    eV = np.sqrt(eL2**2 + eH1s**2)
    enod = np.abs(U - u_exact(X, Y).ravel(order="F")).max()
    return dict(N=N, h=h, nint=int(inn.sum()), eV=eV, eL2=eL2, enod=enod,
                K=K, KII=KII, U=U, X=X, Y=Y, inn=inn, rhs=rhs, b=b)


# ---------------------------------------------------------------- checks
def check_stencil():
    """every interior row of K is (1/3)[8 at the node, -1 at the 8 neighbours],
    and the row sums of K vanish (constants are in the kernel)"""
    r = solve(6)
    K = r["K"].toarray(); n1 = 7
    S = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/3.0
    for i in range(1, 6):
        for j in range(1, 6):
            row = K[i + j*n1].reshape(n1, n1, order="F")
            assert np.allclose(row[i-1:i+2, j-1:j+2], S)
            assert np.isclose(np.abs(row).sum(), np.abs(S).sum())
    assert np.allclose(K.sum(1), 0)
    KII = r["KII"].toarray()
    assert np.allclose(KII, KII.T) and np.linalg.eigvalsh(KII).min() > 0
    # RHS correction for g = 1: +1 next to an edge, +5/3 next to a corner
    corr = (r["rhs"] - r["b"][r["inn"]]).reshape(5, 5, order="F")
    assert np.isclose(corr[0, 0], 5/3) and np.isclose(corr[2, 0], 1.0)
    assert np.isclose(corr[2, 2], 0.0)
    print("stencil, symmetry, positive definiteness, RHS lift: OK")


def check_data():
    import sympy as s
    x, y = s.symbols("x y", real=True)
    u = s.exp(s.sin(s.pi*x)*s.sin(s.pi*y))
    fs = s.pi**2*s.exp(s.sin(s.pi*x)*s.sin(s.pi*y))*(
        2*s.sin(s.pi*x)*s.sin(s.pi*y) + 2*s.cos(s.pi*x)**2*s.cos(s.pi*y)**2
        - s.cos(s.pi*x)**2 - s.cos(s.pi*y)**2)
    res = s.simplify(-s.diff(u, x, 2) - s.diff(u, y, 2) - fs)
    assert res == 0
    assert all(s.simplify(u.subs(x, v)) == 1 for v in (-1, 1))
    assert all(s.simplify(u.subs(y, v)) == 1 for v in (-1, 1))
    print("-Lap u - f = %s ; u = 1 on all four sides: OK" % res)


if __name__ == "__main__":
    check_data()
    check_stencil()
    Ns = [4, 8, 16, 32, 64, 128, 256]
    res = {}
    for load in ("gauss", "trap"):
        res[load] = [solve(N, load) for N in Ns]
        print("\nload =", load)
        print("%5s %9s %7s %12s %6s %12s %6s %11s"
              % ("N", "h", "unknown", "||e||_V", "rate", "||e||_L2", "rate",
                 "max nodal"))
        prev = None
        for r in res[load]:
            rV = rL = "  --"
            if prev:
                rV = "%6.2f" % np.log2(prev["eV"]/r["eV"])
                rL = "%6.2f" % np.log2(prev["eL2"]/r["eL2"])
            print("%5d %9.5f %7d %12.4e %6s %12.4e %6s %11.3e"
                  % (r["N"], r["h"], r["nint"], r["eV"], rV, r["eL2"], rL,
                     r["enod"]))
            prev = r

    def sci(v):
        m, e = ("%.4e" % v).split("e")
        return "%s\\!\\times\\!10^{%d}" % (m, int(e))

    rows = res["gauss"]
    with open("hw4_p4d_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{@{}rrrlclc@{}}\n\\toprule\n")
        fh.write("$N$ & $h$ & unknowns & $\\norm{e}_{V}$ & rate & "
                 "$\\norm{e}_{L^{2}(\\Omega)}$ & rate\\\\\n\\midrule\n")
        prev = None
        for r in rows:
            rV = rL = "---"
            if prev:
                rV = "%.2f" % np.log2(prev["eV"]/r["eV"])
                rL = "%.2f" % np.log2(prev["eL2"]/r["eL2"])
            fh.write("%d & $%.5f$ & %d & $%s$ & %s & $%s$ & %s\\\\\n"
                     % (r["N"], r["h"], r["nint"], sci(r["eV"]), rV,
                        sci(r["eL2"]), rL))
            prev = r
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nwrote hw4_p4d_table.tex")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.1))
    r = solve(32)
    pc = ax[0].pcolormesh(r["X"], r["Y"], r["U"].reshape(33, 33, order="F"),
                          shading="gouraud", cmap="inferno")
    ax[0].contour(r["X"], r["Y"], u_exact(r["X"], r["Y"]), levels=10,
                  colors="w", linewidths=0.6)
    fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.03)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("$U$, $N=32$ (colour); exact $u$ (white contours)",
                    fontsize=10)
    hs = np.array([q["h"] for q in rows])
    eV = np.array([q["eV"] for q in rows]); eL = np.array([q["eL2"] for q in rows])
    ax[1].loglog(hs, eV, "o-", color="#1f4e9c",
                 label=r"$\|e\|_V$  (last slope %.2f)" % np.log2(eV[-2]/eV[-1]))
    ax[1].loglog(hs, eL, "s-", color="#b5651d",
                 label=r"$\|e\|_{L^2}$  (last slope %.2f)" % np.log2(eL[-2]/eL[-1]))
    ax[1].loglog(hs, 0.6*eV[-1]*hs/hs[-1], "k--", lw=0.8, label="$h$")
    ax[1].loglog(hs, 0.6*eL[-1]*(hs/hs[-1])**2, "k:", lw=0.8, label="$h^2$")
    ax[1].set_xlabel("$h$"); ax[1].set_ylabel("error"); ax[1].invert_xaxis()
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("convergence under uniform refinement", fontsize=10)
    fig.tight_layout(); fig.savefig("hw4_p4d_figure.pdf")
    print("wrote hw4_p4d_figure.pdf")
