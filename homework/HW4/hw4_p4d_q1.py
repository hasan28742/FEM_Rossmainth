"""
HW4 Problem 4(d) -- bilinear square elements of Problem 4(c) on [-1,1]^2.

Equation numbers (4c.x), (4d.x) refer to hw4_submission_final.pdf.

    -Laplace(u) = f  in Omega = (-1,1)^2,     u = g = 1  on dOmega
    exact:  u = exp(sin(pi x) sin(pi y))                        (statement of 4(d))

Uniform N x N grid, h = 2/N, nodes (x_i, y_j) = (-1 + i h, -1 + j h); the square
Omega_ij = [x_i, x_{i+1}] x [y_j, y_{j+1}] has corners p = 1..4 counter-clockwise
from (x_i, y_j), as in Figure 3.

What is computed, and where it is derived:
  element matrix k^e ............................................ (4c.12)
  assembly of A from k^e, row = stencil ............ Figures 4, 5; (4c.13)
  load b_ij = h^2 f(x_i, y_j), trapezoidal rule .................. (4c.14)
  boundary values g moved to the right-hand side: A U = b + r .. (4c.15), (4c.16)
  hat functions and their gradients on a square ......... (4c.3), (4c.6)
  error norms ..................................................... (4d.2)
  trapezoidal weights and the rule on one square ......... (4d.3), (4d.4)
  ||e||_L2^2 as a sum, the gradient part as a sum ........ (4d.5), (4d.6)
  consistency -Lap u = f, u = g on the boundary ................... (4d.1)

Only lecture tools are used: trapezoidal rule (Lecture 4), boundary lift
(Lecture 5), A_ij = B(phi_i, phi_j) (Lecture 13).

Writes hw4_p4d_table.tex and hw4_p4d_figure.pdf.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PI = np.pi
M = 16          # trapezoidal sub-intervals per side of a square, M in (4d.3)-(4d.6)


# ---------------------------------------------------------------- data (4(d))
def u_exact(x, y):
    return np.exp(np.sin(PI*x)*np.sin(PI*y))


def grad_u(x, y):
    """exact du/dx, du/dy, as below (4d.6)"""
    u = u_exact(x, y)
    return (u*PI*np.cos(PI*x)*np.sin(PI*y), u*PI*np.sin(PI*x)*np.cos(PI*y))


def f(x, y):
    sx, sy, cx, cy = np.sin(PI*x), np.sin(PI*y), np.cos(PI*x), np.cos(PI*y)
    return PI**2*np.exp(sx*sy)*(2*sx*sy + 2*cx**2*cy**2 - cx**2 - cy**2)


def g(x, y):
    return np.ones_like(x)


# ------------------------------------------------- the element of 4(c)
k_e = np.array([[4, -1, -2, -1],
                [-1, 4, -1, -2],
                [-2, -1, 4, -1],
                [-1, -2, -1, 4]], float)/6.0                   # (4c.12)


def hats(x, y, x_i, y_j, h):
    """Hat functions phi_1..phi_4 of the corners of Omega_ij at (x, y), (4c.3),
    and their partial derivatives d/dx, d/dy, (4c.6)."""
    x_ip1, y_jp1 = x_i + h, y_j + h
    phi = [(x_ip1 - x)*(y_jp1 - y), (x - x_i)*(y_jp1 - y),
           (x - x_i)*(y - y_j), (x_ip1 - x)*(y - y_j)]           # (4c.3)
    dphi_dx = [-(y_jp1 - y), (y_jp1 - y), (y - y_j), -(y - y_j)]   # (4c.6)
    dphi_dy = [-(x_ip1 - x), -(x - x_i), (x - x_i), (x_ip1 - x)]   # (4c.6)
    return ([v/h**2 for v in phi], [v/h**2 for v in dphi_dx],
            [v/h**2 for v in dphi_dy])


def solve(N):
    h = 2.0/N
    n1 = N + 1
    xs = -1.0 + h*np.arange(n1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")           # X[i,j] = x_i, Y[i,j] = y_j
    node = lambda i, j: i + j*n1                         # global number of node (i,j)

    # connectivity t: global nodes at corners 1..4 of each square (Figure 5)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    I = I.ravel(); J = J.ravel()
    t = np.stack([node(I, J), node(I + 1, J), node(I + 1, J + 1),
                  node(I, J + 1)], axis=1)               # (N^2, 4)

    # assembly: add k_e of every square into A (Figures 4, 5); each interior
    # row of A is the stencil (4c.13)
    rows = np.repeat(t, 4, axis=1).ravel()
    cols = np.tile(t, (1, 4)).ravel()
    vals = np.tile(k_e.ravel(), N*N)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n1*n1, n1*n1))

    # load, trapezoidal rule: b_ij = h^2 f(x_i, y_j)              (4c.14)
    b = h*h*f(X, Y).ravel(order="F")

    # boundary values U = g go to the right-hand side: A U = b + r,
    # r = -A_IB g                                                (4c.15), (4c.16)
    onb = np.zeros((n1, n1), bool)
    onb[0, :] = onb[-1, :] = onb[:, 0] = onb[:, -1] = True
    onb = onb.ravel(order="F")
    inn = ~onb
    gB = g(X, Y).ravel(order="F")[onb]
    A_II = A[inn][:, inn]                                # A of (4c.15)
    rhs = b[inn] - A[inn][:, onb] @ gB                   # b + r
    U = np.empty(n1*n1)
    U[onb] = gB
    U[inn] = spla.spsolve(A_II.tocsc(), rhs)

    # errors (4d.2), evaluated as the sums (4d.5) and (4d.6)
    U_p = U[t]                                           # (N^2, 4): U_1..U_4 of each square
    x_i = X.ravel(order="F")[t[:, 0]]                    # left side of each square
    y_j = Y.ravel(order="F")[t[:, 0]]                    # bottom side of each square
    w = np.full(M + 1, h/M); w[[0, M]] = h/(2*M)         # weights w_a, (4d.3)
    sum_L2 = sum_grad = 0.0
    for a in range(M + 1):
        for bb in range(M + 1):
            x = x_i + a*h/M                              # x_{i,a}
            y = y_j + bb*h/M                             # y_{j,b}
            phi, dphi_dx, dphi_dy = hats(x, y, x_i, y_j, h)
            U_ab = sum(U_p[:, p]*phi[p] for p in range(4))        # U = sum U_p phi_p
            Ux_ab = sum(U_p[:, p]*dphi_dx[p] for p in range(4))   # dU/dx
            Uy_ab = sum(U_p[:, p]*dphi_dy[p] for p in range(4))   # dU/dy
            ux, uy = grad_u(x, y)
            sum_L2 += w[a]*w[bb]*np.sum((u_exact(x, y) - U_ab)**2)             # (4d.5)
            sum_grad += w[a]*w[bb]*np.sum((ux - Ux_ab)**2 + (uy - Uy_ab)**2)   # (4d.6)
    eL2 = np.sqrt(sum_L2)                                # ||e||_L2, (4d.5)
    eV = np.sqrt(sum_L2 + sum_grad)                      # ||e||_V = (4d.5) + (4d.6)
    enod = np.abs(U - u_exact(X, Y).ravel(order="F")).max()   # printed only, not used
    return dict(N=N, h=h, nint=int(inn.sum()), eV=eV, eL2=eL2, enod=enod,
                A=A, A_II=A_II, U=U, X=X, Y=Y, inn=inn, rhs=rhs, b=b)


# ---------------------------------------------------------------- checks
def check_stencil():
    """every interior row of A is the stencil (4c.13), 1/3 [8 at the node, -1 at
    the 8 neighbours]; row sums of A vanish; A of (4c.15) is symmetric positive
    definite; the boundary terms are r of (4c.16)"""
    r = solve(6)
    A = r["A"].toarray(); n1 = 7
    S = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/3.0
    for i in range(1, 6):
        for j in range(1, 6):
            row = A[i + j*n1].reshape(n1, n1, order="F")
            assert np.allclose(row[i-1:i+2, j-1:j+2], S)
            assert np.isclose(np.abs(row).sum(), np.abs(S).sum())
    assert np.allclose(A.sum(1), 0)
    A_II = r["A_II"].toarray()
    assert np.allclose(A_II, A_II.T) and np.linalg.eigvalsh(A_II).min() > 0
    # r for g = 1: 5/3 next to a corner, 1 next to a side, 0 otherwise (4c.16)
    rr = (r["rhs"] - r["b"][r["inn"]]).reshape(5, 5, order="F")
    assert np.isclose(rr[0, 0], 5/3) and np.isclose(rr[2, 0], 1.0)
    assert np.isclose(rr[2, 2], 0.0)
    print("stencil (4c.13), symmetry and positive definiteness, r of (4c.16): OK")


def check_data():
    """-Lap u = f and u = g = 1 on the boundary, (4d.1)"""
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
    print("(4d.1): -Lap u - f = %s ; u = 1 on all four sides: OK" % res)


if __name__ == "__main__":
    check_data()
    check_stencil()
    Ns = [4, 8, 16, 32, 64, 128, 256]
    rows = [solve(N) for N in Ns]
    # observed rate log2(E(h)/E(h/2)), as defined in the text after (4d.6)
    print("\n%5s %9s %7s %12s %6s %12s %6s %11s"
          % ("N", "h", "unknown", "||e||_V", "rate", "||e||_L2", "rate",
             "max nodal"))
    prev = None
    for r in rows:
        rV = rL = "  --"
        if prev:
            rV = "%6.2f" % np.log2(prev["eV"]/r["eV"])
            rL = "%6.2f" % np.log2(prev["eL2"]/r["eL2"])
        print("%5d %9.5f %7d %12.4e %6s %12.4e %6s %11.3e"
              % (r["N"], r["h"], r["nint"], r["eV"], rV, r["eL2"], rL, r["enod"]))
        prev = r

    def sci(v):
        m, e = ("%.4e" % v).split("e")
        return "%s\\!\\times\\!10^{%d}" % (m, int(e))

    # Table (tab:p4d) of the document
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

    # Figure (fig:p4d) of the document
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C_V, C_L2 = "#2a78d6", "#eb6834"                     # blue, orange
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.1))

    # left: FE solution U at N = 32, with contours of the exact u
    r = solve(32)
    pc = ax[0].pcolormesh(r["X"], r["Y"], r["U"].reshape(33, 33, order="F"),
                          shading="gouraud", cmap="Blues")
    ax[0].contour(r["X"], r["Y"], u_exact(r["X"], r["Y"]), levels=8,
                  colors="0.15", linewidths=0.7)
    cb = fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label("$U$")
    ax[0].legend(handles=[Line2D([], [], color="0.15", lw=0.7, label="exact $u$ (contours)")],
                 loc="upper right", fontsize=8, framealpha=0.9)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("FE solution $U$, $N=32$", fontsize=10)

    # right: the two errors of Table (tab:p4d) against h, with slopes 1 and 2
    hs = np.array([q["h"] for q in rows])
    eV = np.array([q["eV"] for q in rows]); eL = np.array([q["eL2"] for q in rows])
    ax[1].loglog(hs, eV, "o-", color=C_V, lw=1.5, ms=5, label=r"$\|e\|_V$")
    ax[1].loglog(hs, eL, "s-", color=C_L2, lw=1.5, ms=5, label=r"$\|e\|_{L^2(\Omega)}$")
    ax[1].loglog(hs, 0.6*eV[-1]*hs/hs[-1], "--", color="0.45", lw=1.0,
                 label=r"$C\,h$ (slope 1)")
    ax[1].loglog(hs, 0.6*eL[-1]*(hs/hs[-1])**2, ":", color="0.45", lw=1.2,
                 label=r"$C\,h^2$ (slope 2)")
    ax[1].set_xlabel("$h$"); ax[1].set_ylabel("error")
    ax[1].grid(True, which="major", color="0.88", lw=0.6)
    ax[1].legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax[1].set_title("Convergence", fontsize=10)
    fig.tight_layout(); fig.savefig("hw4_p4d_figure.pdf")
    print("wrote hw4_p4d_figure.pdf")
