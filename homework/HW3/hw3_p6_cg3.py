"""
HW3 Problem 6 -- CG(3) (cubic Hermite) for the cantilever beam

    u'''' = f  on (0,1),   u(0)=u'(0)=0,   u''(1)=u'''(1)=0

on a uniform mesh, with the manufactured data given in the problem statement.
Reports the convergence rate in the energy norm  ||e||_E = ||(u-U)''||
and in the L2 norm  ||e|| = ||u-U||.

Mesh convention follows the lectures: M interior nodes, NE = M+1 elements,
h = 1/NE.  Node 0 is clamped (both DOFs removed); node NE is free (both kept),
so dim V_h = 2*NE.
"""
import numpy as np

PI = np.pi

# ---------------------------------------------------------------- data
def p1(x):
    return -64*PI*(18 - 6*PI**4*x**2 + 18*x**3*PI**2 + 2*PI**4*x - 36*PI**2*x**2
                   - 27*x + 12*PI**2*x + 4*PI**4*x**3 + 3*PI**2)

def p2(x):
    return (216 - 816*PI**2 - 96*PI**4 + 32*PI**6*x**2 - 384*PI**4*x**3
            - 64*PI**6*x**3 - 288*PI**4*x**2 + 3456*PI**2*x + 576*PI**4*x
            - 2592*PI**2*x**2 + 144*PI**4*x**4 + 32*PI**6*x**4)

def f(x):
    return p1(x)*np.cos(2*PI*x) + p2(x)*np.sin(2*PI*x)

def u_exact(x):
    return ((18+2*PI**2)*x**2 + (-24-4*PI**2)*x**3 + (9+2*PI**2)*x**4)*np.sin(2*PI*x)

def d2u_exact(x):
    """u'' of the manufactured solution (product rule, done once)."""
    g   = (18+2*PI**2)*x**2 + (-24-4*PI**2)*x**3 + (9+2*PI**2)*x**4
    dg  = 2*(18+2*PI**2)*x + 3*(-24-4*PI**2)*x**2 + 4*(9+2*PI**2)*x**3
    d2g = 2*(18+2*PI**2) + 6*(-24-4*PI**2)*x + 12*(9+2*PI**2)*x**2
    s, c = np.sin(2*PI*x), np.cos(2*PI*x)
    return d2g*s + 2*dg*(2*PI)*c - g*(2*PI)**2*s

# ------------------------------------------------- Hermite shape functions
def N(t, h):
    """values of N1..N4 at local coordinate t in [0,h]"""
    s = t/h
    return np.array([1 - 3*s**2 + 2*s**3,
                     h*(s - 2*s**2 + s**3),
                     3*s**2 - 2*s**3,
                     h*(-s**2 + s**3)])

def d2N(t, h):
    """second derivatives of N1..N4 w.r.t. x"""
    s = t/h
    return np.array([(-6 + 12*s)/h**2,
                     (-4 + 6*s)/h,
                     ( 6 - 12*s)/h**2,
                     (-2 + 6*s)/h])

# ---------------------------------------------------------------- solve
def solve(NE, nq=12):
    h = 1.0/NE
    xs = np.linspace(0.0, 1.0, NE+1)
    gp, gw = np.polynomial.legendre.leggauss(nq)
    tq = 0.5*h*(gp + 1.0)          # quadrature points on [0,h]
    wq = 0.5*h*gw

    ndof = 2*(NE+1)                # node i -> dofs 2i (deflection), 2i+1 (slope)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # element Hermite stiffness, EI = 1
    ke = np.array([[ 12,  6*h, -12,  6*h],
                   [6*h, 4*h*h, -6*h, 2*h*h],
                   [-12, -6*h,  12, -6*h],
                   [6*h, 2*h*h, -6*h, 4*h*h]], float)/h**3

    for e in range(NE):
        loc = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(loc, loc)] += ke
        # consistent load by Gauss quadrature: no interpolation of f
        fq = f(xs[e] + tq)
        for a in range(4):
            F[loc[a]] += np.sum(wq * fq * N(tq, h)[a])

    free = np.arange(2, ndof)      # drop node 0: clamped (deflection and slope)
    d = np.zeros(ndof)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    # ------------------------------------------------------- errors
    gpE, gwE = np.polynomial.legendre.leggauss(20)   # finer rule for the errors
    tE = 0.5*h*(gpE + 1.0)
    wE = 0.5*h*gwE
    e_E2 = 0.0
    e_L2 = 0.0
    for e in range(NE):
        loc = [2*e, 2*e+1, 2*e+2, 2*e+3]
        de = d[loc]
        xq = xs[e] + tE
        Uh   = de @ N(tE, h)
        d2Uh = de @ d2N(tE, h)
        e_L2 += np.sum(wE*(u_exact(xq) - Uh)**2)
        e_E2 += np.sum(wE*(d2u_exact(xq) - d2Uh)**2)
    return h, np.sqrt(e_E2), np.sqrt(e_L2), d

# ---------------------------------------------------------------- table
if __name__ == "__main__":
    print(__doc__)
    NEs = [4, 8, 16, 32, 64, 128]
    rows = []
    prev = None
    for NE in NEs:
        h, eE, eL, _ = solve(NE)
        if prev is None:
            rE = rL = None
        else:
            rE = np.log2(prev[0]/eE)
            rL = np.log2(prev[1]/eL)
        rows.append((NE, NE-1, h, eE, rE, eL, rL))
        prev = (eE, eL)

    hdr = f"{'elems':>6} {'M':>5} {'h':>10} {'||e||_E':>13} {'rate':>7} {'||e||_L2':>13} {'rate':>7}"
    print(hdr); print("-"*len(hdr))
    for NE, M, h, eE, rE, eL, rL in rows:
        sE = "  --  " if rE is None else f"{rE:6.2f}"
        sL = "  --  " if rL is None else f"{rL:6.2f}"
        print(f"{NE:6d} {M:5d} {h:10.5f} {eE:13.4e} {sE:>7} {eL:13.4e} {sL:>7}")

    def sci(v):
        """format a float as  m.mmmm x 10^{e}  for LaTeX math mode"""
        m, ex = f"{v:.4e}".split("e")
        return f"{m}\\!\\times\\!10^{{{int(ex)}}}"

    # LaTeX table for the writeup
    with open("hw3_p6_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{@{}rrrlcrc@{}}\n\\toprule\n")
        fh.write("elements & $M$ & $h$ & $\\norm{e}_{E}$ & rate & "
                 "$\\norm{e}_{L^{2}}$ & rate\\\\\n\\midrule\n")
        for NE, M, h, eE, rE, eL, rL in rows:
            sE = "---" if rE is None else f"{rE:.2f}"
            sL = "---" if rL is None else f"{rL:.2f}"
            fh.write(f"{NE} & {M} & ${h:.5f}$ & ${sci(eE)}$ & {sE} & "
                     f"${sci(eL)}$ & {sL}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nwrote hw3_p6_table.tex")

    # convergence plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs  = np.array([r[2] for r in rows])
    eEs = np.array([r[3] for r in rows])
    eLs = np.array([r[5] for r in rows])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(hs, eEs, "o-", color="#1f4e9c", label=r"$\|e\|_E$  (slope %.2f)" % rows[-1][4])
    ax.loglog(hs, eLs, "s-", color="#b5651d", label=r"$\|e\|_{L^2}$  (slope %.2f)" % rows[-1][6])
    ax.set_xlabel("$h$"); ax.set_ylabel("error")
    ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("hw3_p6_convergence.pdf")
    print("wrote hw3_p6_convergence.pdf")
