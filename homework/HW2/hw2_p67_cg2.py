#!/usr/bin/env python3
"""
MATH 6660 -- Homework 2, Problems 6 and 7.

CG(2) (piecewise quadratic) for the Dirichlet two-point BVP

    -u'' = f(x)   on (0,1),      u(0) = u(1) = 0

Problem 6 is the implementation; Problem 7 applies it to

    u(x) = exp(1 - cos(2 pi x) - sin(2 pi x)) - 1
    f(x) = 4 pi^2 exp(1 - cos(2 pi x) - sin(2 pi x))
           * [ sin(4 pi x) - cos(2 pi x) - sin(2 pi x) - 1 ]

Numbering follows Lecture 10: half-integer node indices k = 0,...,2M+2 with
x_{k/2} = k h / 2, so k even is a vertex and k odd is a midside node.  The two
Dirichlet ends k = 0 and k = 2M+2 are eliminated, leaving 2M+1 unknowns.

Outputs
-------
  hw2_p67_solution.pdf / .png     exact vs CG(2) solution, and the error
  hw2_p67_convergence.pdf / .png  log-log convergence, both norms
  hw2_p67_table.tex               the convergence table, \\input-able
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_DOM, B_DOM = 0.0, 1.0
TWOPI = 2.0 * np.pi


# ----------------------------------------------------------------------
# problem data  (Problem 7)
# ----------------------------------------------------------------------
def _g(x):
    return 1.0 - np.cos(TWOPI * x) - np.sin(TWOPI * x)


def u_exact(x):
    return np.exp(_g(x)) - 1.0


def du_exact(x):
    # u' = e^g g',   g' = 2 pi ( sin 2pi x - cos 2pi x )
    gp = TWOPI * (np.sin(TWOPI * x) - np.cos(TWOPI * x))
    return np.exp(_g(x)) * gp


def d2u_exact(x):
    # u'' = e^g ( g'^2 + g'' ),  g'^2 = 4 pi^2 (1 - sin 4pi x)
    gp2 = 4.0 * np.pi**2 * (1.0 - np.sin(2.0 * TWOPI * x))
    gpp = 4.0 * np.pi**2 * (np.cos(TWOPI * x) + np.sin(TWOPI * x))
    return np.exp(_g(x)) * (gp2 + gpp)


def f_fun(x):
    return (4.0 * np.pi**2 * np.exp(_g(x))
            * (np.sin(2.0 * TWOPI * x) - np.cos(TWOPI * x)
               - np.sin(TWOPI * x) - 1.0))


def check_manufactured():
    x = np.linspace(A_DOM, B_DOM, 20001)
    resid = np.max(np.abs(-d2u_exact(x) - f_fun(x)))
    bc = abs(u_exact(0.0)) + abs(u_exact(1.0))
    return resid, bc


# ----------------------------------------------------------------------
# reference element [0,1]:  quadratic Lagrange basis
#   N0 = 2(xi-1/2)(xi-1)   vertex at xi=0
#   N1 = 4 xi (1-xi)       midside at xi=1/2
#   N2 = 2 xi (xi-1/2)     vertex at xi=1
# ----------------------------------------------------------------------
def shape(xi):
    N = np.stack([2.0 * (xi - 0.5) * (xi - 1.0),
                  4.0 * xi * (1.0 - xi),
                  2.0 * xi * (xi - 0.5)])
    dN = np.stack([4.0 * xi - 3.0,
                   4.0 - 8.0 * xi,
                   4.0 * xi - 1.0])
    return N, dN          # dN is d/dxi; divide by h for d/dx


NQ = 8
_gp, _gw = np.polynomial.legendre.leggauss(NQ)
XI = 0.5 * (_gp + 1.0)
WQ = 0.5 * _gw
NSH, DNSH = shape(XI)                      # (3, NQ)


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------
def assemble(M):
    """
    Returns (A_reduced, b_reduced, h, ndof_full).
    Full half-integer numbering k = 0..2M+2;  k=0 and k=2M+2 are Dirichlet.
    """
    nel = M + 1
    h = (B_DOM - A_DOM) / nel
    nfull = 2 * nel + 1                    # = 2M+3 nodes

    # element stiffness is the same on every element (a == 1, uniform h)
    Ke = (DNSH * WQ) @ DNSH.T / h          # (3,3);  = (1/h) int dN_a dN_b dxi

    # element loads: quadrature points of all elements at once
    xl = A_DOM + h * np.arange(nel)
    xq = xl[:, None] + h * XI[None, :]     # (nel, NQ)
    fq = f_fun(xq)
    be = (fq * (WQ * h)) @ NSH.T           # (nel, 3)

    # local -> global:  element e owns half-indices 2e, 2e+1, 2e+2
    loc = 2 * np.arange(nel)[:, None] + np.array([0, 1, 2])[None, :]   # (nel,3)

    rows = np.repeat(loc, 3, axis=1).ravel()
    cols = np.tile(loc, (1, 3)).ravel()
    vals = np.tile(Ke.ravel(), nel)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(nfull, nfull)).tocsr()

    b = np.zeros(nfull)
    np.add.at(b, loc.ravel(), be.ravel())

    # homogeneous Dirichlet: drop the two end rows/columns
    keep = np.arange(1, nfull - 1)
    return A[keep][:, keep].tocsr(), b[keep], h, nfull


def solve(M):
    Ar, br, h, nfull = assemble(M)
    xi, info = spla.cg(Ar, br, rtol=1e-13, atol=0.0, maxiter=50000)
    if info != 0:
        xi = spla.spsolve(Ar.tocsc(), br)
        info = -1
    full = np.zeros(nfull)
    full[1:-1] = xi                        # reinstate the zero end values
    return full, h, info, Ar


# ----------------------------------------------------------------------
# verification against the hand-computed matrix of Lecture 10
# ----------------------------------------------------------------------
def check_matrix(M=4):
    """
    Lecture 10 (a == 1, uniform h) gives, for the 2M+1 unknowns,
        A_ii   = 16/(3h)   i odd  (midside)      A_{i,i+-1} = -8/(3h)
        A_ii   = 14/(3h)   i even (vertex)       A_{i,i+-2} =  1/(3h)
    so 3h*A should be an integer pentadiagonal matrix.
    """
    Ar, _, h, _ = assemble(M)
    S = np.asarray((3.0 * h * Ar).todense())
    S = np.round(S, 10)
    n = S.shape[0]
    expect = np.zeros((n, n))
    for i in range(n):
        k = i + 1                          # half-integer index of this unknown
        expect[i, i] = 16.0 if k % 2 == 1 else 14.0
        if i + 1 < n:
            expect[i, i + 1] = expect[i + 1, i] = -8.0
        if k % 2 == 0 and i + 2 < n:       # vertex-to-vertex, two apart
            expect[i, i + 2] = expect[i + 2, i] = 1.0
    ok = np.allclose(S, expect)
    return ok, S, expect


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
def errors(full, h):
    """L2 and energy norm of u-U.  Energy here is ||v||_E^2 = int (v')^2 dx."""
    nel = (len(full) - 1) // 2
    loc = 2 * np.arange(nel)[:, None] + np.array([0, 1, 2])[None, :]
    coef = full[loc]                        # (nel, 3)

    xl = A_DOM + h * np.arange(nel)
    xq = xl[:, None] + h * XI[None, :]
    w = WQ * h

    Uq = coef @ NSH                         # (nel, NQ)
    dUq = (coef @ DNSH) / h

    eq = u_exact(xq) - Uq
    deq = du_exact(xq) - dUq
    l2 = np.sqrt(np.sum(eq**2 * w))
    en = np.sqrt(np.sum(deq**2 * w))
    return l2, en


def eval_U(full, h, x):
    """Evaluate the piecewise quadratic U at arbitrary points x."""
    nel = (len(full) - 1) // 2
    e = np.clip(np.floor((x - A_DOM) / h).astype(int), 0, nel - 1)
    xi = (x - (A_DOM + e * h)) / h
    N, _ = shape(xi)                        # (3, npts)
    idx = 2 * e
    return full[idx] * N[0] + full[idx + 1] * N[1] + full[idx + 2] * N[2]


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figure_solution(M=12, stem="hw2_p67_solution"):
    full, h, *_ = solve(M)
    nel = M + 1
    xf = np.linspace(A_DOM, B_DOM, 4001)
    Uf = eval_U(full, h, xf)

    xv = A_DOM + h * np.arange(nel + 1)          # vertices
    xm = A_DOM + h * (np.arange(nel) + 0.5)      # midsides

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                           gridspec_kw={"height_ratios": [2.1, 1.0]})

    ax[0].plot(xf, u_exact(xf), "-", lw=2.4, color="0.35", label="exact $u(x)$")
    ax[0].plot(xf, Uf, "-", lw=1.5, color="#1f6f8b",
               label=rf"CG(2), $M={M}$  ({nel} elements, {2*M+1} unknowns)")
    ax[0].plot(xv, eval_U(full, h, xv), "o", ms=5.0, color="#1f6f8b",
               mfc="white", mew=1.4, label="vertex nodes")
    ax[0].plot(xm, eval_U(full, h, xm), "^", ms=5.0, color="#b5651d",
               mfc="white", mew=1.4, label="midside nodes")
    for xe in xv:
        ax[0].axvline(xe, color="0.85", lw=0.7, zorder=0)
    ax[0].set_ylabel(r"$u$")
    ax[0].legend(loc="upper left", fontsize=8.5, framealpha=1.0)
    ax[0].set_title("Problems 6--7 — CG(2), Dirichlet BCs", fontsize=11)
    ax[0].grid(alpha=0.2)

    ax[1].plot(xf, u_exact(xf) - Uf, "-", lw=1.4, color="#b23a3a")
    ax[1].axhline(0.0, color="0.6", lw=0.8)
    for xe in xv:
        ax[1].axvline(xe, color="0.85", lw=0.7, zorder=0)
    ax[1].set_ylabel(r"$u - U$")
    ax[1].set_xlabel(r"$x$")
    ax[1].grid(alpha=0.2)
    ax[1].set_title("pointwise error — exactly zero at the element vertices, "
                    "not at the midsides", fontsize=9.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_convergence(hs, e_l2, e_en, stem="hw2_p67_convergence"):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.loglog(hs, e_en, "o-", lw=1.8, ms=6, color="#1f6f8b",
              label=r"$\|u-U\|_E$   (energy)")
    ax.loglog(hs, e_l2, "s-", lw=1.8, ms=6, color="#b23a3a",
              label=r"$\|u-U\|_{L^2}$")

    ax.set_xlabel(r"$h = 1/(M+1)$")
    ax.set_ylabel("error")
    ax.set_title("Problem 7 — CG(2) convergence", fontsize=11)
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=9.5, framealpha=1.0)
    ax.invert_xaxis()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def rates(hs, errs):
    r = [np.nan]
    for k in range(1, len(errs)):
        r.append(np.log(errs[k - 1] / errs[k]) / np.log(hs[k - 1] / hs[k]))
    return np.array(r)


# exact P2 element mass matrix, (h/30) * [[4,2,-1],[2,16,2],[-1,2,4]].
# Applied to nodal f this reproduces Lecture 10's interpolated load:
#   midside rows  (h/15)[ f + 8f + f ]
#   vertex  rows  (h/30)[ -f + 2f + 8f + 2f - f ]
MASS_E = np.array([[4.0, 2.0, -1.0],
                   [2.0, 16.0, 2.0],
                   [-1.0, 2.0, 4.0]]) / 30.0


def solve_interp_load(M):
    """Same CG(2) solve, but with Lecture 10's interpolated load instead of Gauss."""
    nel = M + 1
    h = (B_DOM - A_DOM) / nel
    nfull = 2 * nel + 1
    Ke = (DNSH * WQ) @ DNSH.T / h
    loc = 2 * np.arange(nel)[:, None] + np.array([0, 1, 2])[None, :]
    A = sp.coo_matrix((np.tile(Ke.ravel(), nel),
                       (np.repeat(loc, 3, axis=1).ravel(), np.tile(loc, (1, 3)).ravel())),
                      shape=(nfull, nfull)).tocsr()
    fn = f_fun(A_DOM + h * 0.5 * np.arange(nfull))     # f sampled at ALL nodes
    b = np.zeros(nfull)
    for e in range(nel):
        b[loc[e]] += h * (MASS_E @ fn[loc[e]])
    keep = np.arange(1, nfull - 1)
    x = spla.spsolve(A[keep][:, keep].tocsc(), b[keep])
    full = np.zeros(nfull)
    full[1:-1] = x
    return full, h


def interp_load_compare(Ms=(24, 49, 99, 199, 399, 799)):
    """
    How much does Lecture 10's interpolated load change the answer?
    Reported in the Problem 6 note of the write-up.
    """
    def rate(prev, cur):
        return "  --  " if prev is None else f"{np.log(prev / cur) / np.log(2.0):6.2f}"

    print(f"{'M':>5} {'h':>11} | {'L2 gauss':>11} {'L2 interp':>11} {'r_g':>6} {'r_i':>6}"
          f" | {'|dU|_L2':>11} {'rate':>6} {'|dU|_E':>11} {'rate':>6} | {'vert(interp)':>12} {'rate':>6}")
    print("-" * 126)
    pg = pi = pdl = pde = pv = None
    for M in Ms:
        fg, h, _, _ = solve(M)
        fi, _ = solve_interp_load(M)
        lg, eg = errors(fg, h)
        li, ei = errors(fi, h)
        nel = M + 1
        loc = 2 * np.arange(nel)[:, None] + np.array([0, 1, 2])[None, :]
        c = (fi - fg)[loc]
        w = WQ * h
        dl = np.sqrt(np.sum((c @ NSH) ** 2 * w))
        de = np.sqrt(np.sum(((c @ DNSH) / h) ** 2 * w))
        xv = A_DOM + h * np.arange(nel + 1)
        vi = float(np.max(np.abs(u_exact(xv) - eval_U(fi, h, xv))))
        print(f"{M:5d} {h:11.4e} | {lg:11.4e} {li:11.4e} {rate(pg, lg)} {rate(pi, li)}"
              f" | {dl:11.4e} {rate(pdl, dl)} {de:11.4e} {rate(pde, de)}"
              f" | {vi:12.4e} {rate(pv, vi)}")
        pg, pi, pdl, pde, pv = lg, li, dl, de, vi
    print("-> the difference is O(h^4) in both norms, but the L2 column and the first")
    print("   printed rate still move at the coarse end; and vertex exactness is lost.")


def nodal_check(Ms=(12, 24, 49, 99)):
    """CG(2) for -u''=f is exact at the element VERTICES (not at the midsides)."""
    print(f"{'M':>5} {'h':>12} {'max err at vertices':>21} {'max err at midsides':>21}")
    print("-" * 62)
    for M in Ms:
        full, h, _, _ = solve(M)
        nel = M + 1
        xv = A_DOM + h * np.arange(nel + 1)
        xm = A_DOM + h * (np.arange(nel) + 0.5)
        ev = float(np.max(np.abs(u_exact(xv) - eval_U(full, h, xv))))
        em = float(np.max(np.abs(u_exact(xm) - eval_U(full, h, xm))))
        print(f"{M:5d} {h:12.6e} {ev:21.4e} {em:21.4e}")


def main():
    res, bc = check_manufactured()
    print(f"manufactured solution:  max|-u''-f| = {res:.3e}   "
          f"|u(0)|+|u(1)| = {bc:.3e}")

    ok, S, expect = check_matrix(M=4)
    print(f"\nProblem 6 check: 3h*A against Lecture 10 "
          f"(M=4, {S.shape[0]}x{S.shape[0]}):  {'MATCH' if ok else 'MISMATCH'}")
    print("  assembled 3h*A =")
    for row in S.astype(int):
        print("   ", " ".join(f"{v:4d}" for v in row))
    print()

    Ms = [24, 49, 99, 199, 399, 799]
    hs, e_l2, e_en = [], [], []
    print(f"{'M':>5} {'elems':>7} {'unknowns':>9} {'h':>12} "
          f"{'||e||_L2':>13} {'||e||_E':>13} {'cg':>7}")
    print("-" * 74)
    for M in Ms:
        full, h, info, _ = solve(M)
        l2, en = errors(full, h)
        hs.append(h); e_l2.append(l2); e_en.append(en)
        print(f"{M:5d} {M+1:7d} {2*M+1:9d} {h:12.6e} {l2:13.6e} {en:13.6e} "
              f"{'ok' if info == 0 else 'direct':>7}")

    hs = np.array(hs); e_l2 = np.array(e_l2); e_en = np.array(e_en)
    r_l2 = rates(hs, e_l2); r_en = rates(hs, e_en)

    print("\nconvergence table")
    print(f"{'M':>5} {'h':>12} {'||e||_L2':>13} {'rate':>7} "
          f"{'||e||_E':>13} {'rate':>7}")
    print("-" * 62)
    for k, M in enumerate(Ms):
        s2 = "   --  " if np.isnan(r_l2[k]) else f"{r_l2[k]:7.3f}"
        sE = "   --  " if np.isnan(r_en[k]) else f"{r_en[k]:7.3f}"
        print(f"{M:5d} {hs[k]:12.6e} {e_l2[k]:13.6e} {s2} {e_en[k]:13.6e} {sE}")
    print()
    print(f"observed asymptotic rates:  L2 -> {r_l2[-1]:.3f}   "
          f"energy -> {r_en[-1]:.3f}")
    print("theory (Lecture 10)       :  L2 -> 3       energy -> 2")

    # ---- convergence table using LECTURE 10's interpolated load -------------
    hs_l, l2_l, en_l = [], [], []
    for M in Ms:
        full, h = solve_interp_load(M)
        l2, en = errors(full, h)
        hs_l.append(h); l2_l.append(l2); en_l.append(en)
    hs_l = np.array(hs_l); l2_l = np.array(l2_l); en_l = np.array(en_l)
    r2_l, rE_l = rates(hs_l, l2_l), rates(hs_l, en_l)
    print("\nconvergence table -- Lecture 10 interpolated load")
    print(f"{'M':>5} {'h':>12} {'||e||_L2':>13} {'rate':>7} {'||e||_E':>13} {'rate':>7}")
    print("-" * 62)
    for k, M in enumerate(Ms):
        a = "   --  " if np.isnan(r2_l[k]) else f"{r2_l[k]:7.3f}"
        b = "   --  " if np.isnan(rE_l[k]) else f"{rE_l[k]:7.3f}"
        print(f"{M:5d} {hs_l[k]:12.6e} {l2_l[k]:13.6e} {a} {en_l[k]:13.6e} {b}")
    with open("hw2_p67_table_lecture.tex", "w") as fh:
        fh.write("\\begin{tabular}{rccccc}\n\\toprule\n")
        fh.write("$M$ & $h$ & $\\|u-U\\|_{L^2}$ & rate & $\\|u-U\\|_E$ & rate\\\\\n\\midrule\n")
        for k, M in enumerate(Ms):
            a = "---" if np.isnan(r2_l[k]) else f"{r2_l[k]:.2f}"
            b = "---" if np.isnan(rE_l[k]) else f"{rE_l[k]:.2f}"
            fh.write(f"{M} & {hs_l[k]:.4e} & {l2_l[k]:.4e} & {a} & {en_l[k]:.4e} & {b}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open("hw2_p67_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{rccccc}\n\\toprule\n")
        fh.write("$M$ & $h$ & $\\|u-U\\|_{L^2}$ & rate & "
                 "$\\|u-U\\|_E$ & rate\\\\\n\\midrule\n")
        for k, M in enumerate(Ms):
            s2 = "---" if np.isnan(r_l2[k]) else f"{r_l2[k]:.2f}"
            sE = "---" if np.isnan(r_en[k]) else f"{r_en[k]:.2f}"
            fh.write(f"{M} & {hs[k]:.4e} & {e_l2[k]:.4e} & {s2} & "
                     f"{e_en[k]:.4e} & {sE}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    print("\nnodal check -- exact at vertices, not at midsides")
    nodal_check()

    print("\nLecture 10's interpolated load vs 8-point Gauss")
    interp_load_compare()

    figure_solution(M=12)
    figure_convergence(hs, e_l2, e_en)
    print("\nwrote hw2_p67_solution.{pdf,png}, hw2_p67_convergence.{pdf,png}, "
          "hw2_p67_table.tex")


if __name__ == "__main__":
    main()
