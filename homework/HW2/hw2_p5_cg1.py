#!/usr/bin/env python3
"""
MATH 6660 -- Homework 2, Problem 5.

CG(1) for the periodic two-point BVP

    -u'' + q(x) u = f(x)   on (-pi, pi),
    u(-pi) = u(pi),   u'(-pi) = u'(pi),        q > 0

on a uniform mesh, with the manufactured solution

    u(x) = exp(sin x) exp(cos x)
    q(x) = 3 - sin(x) - sin(2x) - cos(x)
    f(x) = 2 exp(sin x) exp(cos x)

Outputs
-------
  hw2_p5_solution.pdf / .png     exact vs CG(1) solution, and the error
  hw2_p5_convergence.pdf / .png  log-log convergence, both norms
  hw2_p5_table.tex               the convergence table, \input-able
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# problem data
# ----------------------------------------------------------------------
A_DOM, B_DOM = -np.pi, np.pi


def u_exact(x):
    return np.exp(np.sin(x)) * np.exp(np.cos(x))


def du_exact(x):
    # u = e^(sin x + cos x)  ->  u' = u * (cos x - sin x)
    return u_exact(x) * (np.cos(x) - np.sin(x))


def d2u_exact(x):
    # u'' = u [ (cos x - sin x)^2 - (sin x + cos x) ]
    c, sn = np.cos(x), np.sin(x)
    return u_exact(x) * ((c - sn) ** 2 - (sn + c))


def q_fun(x):
    return 3.0 - np.sin(x) - np.sin(2.0 * x) - np.cos(x)


def f_fun(x):
    return 2.0 * np.exp(np.sin(x)) * np.exp(np.cos(x))


def check_manufactured():
    """Verify -u'' + q u = f exactly (analytic second derivative)."""
    x = np.linspace(A_DOM, B_DOM, 20001)
    resid = -d2u_exact(x) + q_fun(x) * u_exact(x) - f_fun(x)
    per = abs(u_exact(A_DOM) - u_exact(B_DOM)) + abs(du_exact(A_DOM) - du_exact(B_DOM))
    return np.max(np.abs(resid)), per


# ----------------------------------------------------------------------
# Gauss-Legendre on a reference element [0, 1]
# ----------------------------------------------------------------------
NQ = 6
_gp, _gw = np.polynomial.legendre.leggauss(NQ)
XI = 0.5 * (_gp + 1.0)  # quadrature points on [0,1]
WQ = 0.5 * _gw  # weights   on [0,1]


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------
def assemble(M):
    """
    Periodic CG(1) system.  M is the problem's M, so:
      h  = 2 pi / (M+1)
      N  = M+1  distinct nodes  x_0 ... x_M     (x_{M+1} = pi is identified with x_0)
      N        elements, element e joining node e to node (e+1) mod N
    Returns (A_csr, b, x_nodes, h).
    """
    N = M + 1
    h = (B_DOM - A_DOM) / N
    xn = A_DOM + h * np.arange(N)  # x_0 ... x_M

    # local shape functions on [0,1]:  N0 = 1-xi, N1 = xi
    N0, N1 = 1.0 - XI, XI
    dN0, dN1 = -1.0 / h, 1.0 / h

    # element left endpoints, and quadrature points of every element at once
    xl = xn  # element e starts at node e
    xq = xl[:, None] + h * XI[None, :]  # (N, NQ)
    qq = q_fun(xq)
    fq = f_fun(xq)
    w = WQ * h  # physical weights

    # --- element stiffness: constant, same for every element
    k_stiff = np.array([[dN0 * dN0, dN0 * dN1], [dN1 * dN0, dN1 * dN1]]) * h

    # --- element mass weighted by q, per element  (N, 2, 2)
    m00 = (qq * N0 * N0) @ w
    m01 = (qq * N0 * N1) @ w
    m11 = (qq * N1 * N1) @ w

    # --- element load  (N, 2)
    b0 = (fq * N0) @ w
    b1 = (fq * N1) @ w

    # --- scatter into the global system, wrapping the last element round
    lo = np.arange(N)
    hi = (lo + 1) % N  # <-- the periodic wrap

    rows = np.concatenate([lo, lo, hi, hi])
    cols = np.concatenate([lo, hi, lo, hi])
    vals = np.concatenate(
        [
            k_stiff[0, 0] + m00,
            k_stiff[0, 1] + m01,
            k_stiff[1, 0] + m01,
            k_stiff[1, 1] + m11,
        ]
    )
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    b = np.zeros(N)
    np.add.at(b, lo, b0)
    np.add.at(b, hi, b1)

    return A, b, xn, h


def solve(M):
    A, b, xn, h = assemble(M)
    xi, info = spla.cg(A, b, rtol=1e-13, atol=0.0, maxiter=20000)
    if info != 0:  # fall back so a stalled CG never silently poisons a rate
        xi = spla.spsolve(A.tocsc(), b)
        info = -1
    return xi, xn, h, A, b, info


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
def errors(xi, xn, h):
    """L2 and energy norm of u - U, by element Gauss quadrature."""
    N = len(xn)
    lo = np.arange(N)
    hi = (lo + 1) % N

    xq = xn[:, None] + h * XI[None, :]  # (N, NQ)
    w = WQ * h

    # U and U' on each element from the two nodal values
    a0 = xi[lo][:, None]
    a1 = xi[hi][:, None]
    Uq = a0 * (1.0 - XI)[None, :] + a1 * XI[None, :]
    dUq = (a1 - a0) / h  # constant per element, broadcasts

    eq = u_exact(xq) - Uq
    deq = du_exact(xq) - dUq

    l2sq = float((eq**2) @ w @ np.ones(1)) if False else float(np.sum((eq**2) * w))
    ensq = float(np.sum((deq**2 + q_fun(xq) * eq**2) * w))
    return np.sqrt(l2sq), np.sqrt(ensq)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figure_solution(M=24, stem="hw2_p5_solution"):
    xi, xn, h, *_ = solve(M)

    # close the loop for plotting: repeat the first node at x = pi
    xp = np.append(xn, B_DOM)
    Up = np.append(xi, xi[0])

    xf = np.linspace(A_DOM, B_DOM, 2001)

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                           gridspec_kw={"height_ratios": [2.1, 1.0]})

    ax[0].plot(xf, u_exact(xf), "-", lw=2.2, color="0.35",
               label=r"exact $u(x)=e^{\sin x}e^{\cos x}$")
    ax[0].plot(xp, Up, "-", lw=1.6, color="#1f6f8b",
               label=rf"CG(1), $M={M}$  ({M+1} elements)")
    ax[0].plot(xp, Up, "o", ms=4.0, color="#1f6f8b", mfc="white", mew=1.3)
    ax[0].set_ylabel(r"$u$")
    ax[0].legend(loc="upper right", fontsize=9, framealpha=1.0)
    ax[0].set_title("Problem 5 — CG(1) on a periodic mesh", fontsize=11)
    ax[0].grid(alpha=0.25)
    # mark the glued node
    ax[0].axvline(A_DOM, color="#b5651d", lw=1.0, ls=":")
    ax[0].axvline(B_DOM, color="#b5651d", lw=1.0, ls=":")
    ax[0].annotate(r"$x_0 \equiv x_{M+1}$ : one unknown, two ends",
                   xy=(A_DOM, u_exact(A_DOM)), xytext=(-2.9, 2.35),
                   fontsize=8.5, color="#b5651d",
                   arrowprops=dict(arrowstyle="->", color="#b5651d", lw=0.9))

    # pointwise error, sampled densely so the per-element shape is visible
    Uf = np.interp(xf, xp, Up)
    ax[1].plot(xf, u_exact(xf) - Uf, "-", lw=1.4, color="#b23a3a")
    ax[1].axhline(0.0, color="0.6", lw=0.8)
    ax[1].set_ylabel(r"$u - U$")
    ax[1].set_xlabel(r"$x$")
    ax[1].grid(alpha=0.25)
    ax[1].set_title("pointwise error — one bump per element, largest where $u$ is most curved",
                    fontsize=9.5)

    ax[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_convergence(hs, e_l2, e_en, stem="hw2_p5_convergence"):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.loglog(hs, e_en, "o-", lw=1.8, ms=6, color="#1f6f8b",
              label=r"$\|u-U\|_E$   (energy)")
    ax.loglog(hs, e_l2, "s-", lw=1.8, ms=6, color="#b23a3a",
              label=r"$\|u-U\|_{L^2}$")

    # reference slopes, anchored to the finest point of each curve
    ax.set_xlabel(r"$h = 2\pi/(M+1)$")
    ax.set_ylabel("error")
    ax.set_title("Problem 5 — CG(1) convergence", fontsize=11)
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


def main():
    res, per = check_manufactured()
    print(f"manufactured solution:  max residual = {res:.3e}   "
          f"periodicity defect = {per:.3e}")
    print()

    Ms = [24, 49, 99, 199, 399, 799]
    hs, e_l2, e_en = [], [], []

    print(f"{'M':>5} {'N=M+1':>7} {'h':>12} {'||e||_L2':>13} {'||e||_E':>13} "
          f"{'max nodal':>12} {'/||e||_L2':>10} {'cg':>7}")
    print("-" * 88)
    for M in Ms:
        xi, xn, h, A, b, info = solve(M)
        l2, en = errors(xi, xn, h)
        # nodal error: CG(1) is NOT nodally exact here, because of the q u term
        nodal = float(np.max(np.abs(u_exact(xn) - xi)))
        hs.append(h)
        e_l2.append(l2)
        e_en.append(en)
        print(f"{M:5d} {M+1:7d} {h:12.6e} {l2:13.6e} {en:13.6e} "
              f"{nodal:12.4e} {nodal/l2:10.3f} "
              f"{'ok' if info == 0 else 'direct':>7}")

    hs = np.array(hs)
    e_l2 = np.array(e_l2)
    e_en = np.array(e_en)
    r_l2 = rates(hs, e_l2)
    r_en = rates(hs, e_en)

    print()
    print("convergence table")
    print(f"{'M':>5} {'h':>12} {'||e||_L2':>13} {'rate':>7} {'||e||_E':>13} {'rate':>7}")
    print("-" * 62)
    for k, M in enumerate(Ms):
        s2 = "   --  " if np.isnan(r_l2[k]) else f"{r_l2[k]:7.3f}"
        sE = "   --  " if np.isnan(r_en[k]) else f"{r_en[k]:7.3f}"
        print(f"{M:5d} {hs[k]:12.6e} {e_l2[k]:13.6e} {s2} {e_en[k]:13.6e} {sE}")
    print()
    print(f"observed asymptotic rates:  L2 -> {r_l2[-1]:.3f}   energy -> {r_en[-1]:.3f}")
    print("theory (Problems 3 and 4) :  L2 -> 2       energy -> 1")

    # ---- LaTeX table for the writeup
    with open("hw2_p5_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{rccccc}\n\\toprule\n")
        fh.write("$M$ & $h$ & $\\|u-U\\|_{L^2}$ & rate & $\\|u-U\\|_E$ & rate\\\\\n")
        fh.write("\\midrule\n")
        for k, M in enumerate(Ms):
            s2 = "---" if np.isnan(r_l2[k]) else f"{r_l2[k]:.2f}"
            sE = "---" if np.isnan(r_en[k]) else f"{r_en[k]:.2f}"
            fh.write(f"{M} & {hs[k]:.4e} & {e_l2[k]:.4e} & {s2} & "
                     f"{e_en[k]:.4e} & {sE}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    figure_solution(M=24)
    figure_convergence(hs, e_l2, e_en)
    print("\nwrote hw2_p5_solution.{pdf,png}, hw2_p5_convergence.{pdf,png}, "
          "hw2_p5_table.tex")


if __name__ == "__main__":
    main()
