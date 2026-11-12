"""The four combinations of continuous/discontinuous x conforming/non-conforming.

Two axes, four boxes. Rows: is there a gap? Columns: is the gap legal for the
equation being solved? Every box is drawn on the same two elements so the only
thing that changes between boxes is the thing being named.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

INK, OK, BAD, GREY, AMB = "#15284a", "#2e7d32", "#b02020", "#5b6066", "#9a6b10"
FILL = "#dfe6f0"

def quad(ax, pts, fc=FILL, ec=INK, lw=2.2, z=3):
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))
    for p in pts:
        ax.plot(*p, "o", ms=4.6, color=ec, zorder=z+1)

fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.035,
                    wspace=0.13, hspace=0.20)
for ax in axes.ravel():
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def head(ax, n, name, colour, sub):
    ax.text(50, 53, f"{n}.  {name}", fontsize=13.8, color=colour,
            fontweight="bold", ha="center")
    ax.text(50, 47.5, sub, fontsize=12.4, color=GREY, ha="center")

def foot(ax, lines, colour):
    for k, L in enumerate(lines):
        ax.text(50, 9.5 - 5.0*k, L, fontsize=12.2 if k == 0 else 11.4,
                color=colour if k == 0 else GREY, ha="center",
                fontweight="bold" if k == 0 else "normal")

# ---------- 1 : continuous + conforming ----------
ax = axes[0, 0]
head(ax, 1, "CONTINUOUS + CONFORMING", OK, "no gap, and none is wanted")
quad(ax, [(22, 18), (50, 20), (50, 40), (22, 38)])
quad(ax, [(50, 20), (78, 18), (78, 38), (50, 40)])
ax.plot([50, 50], [20, 40], color=OK, lw=4.0, zorder=6)
ax.text(50, 43.5, "one shared edge", fontsize=12.0, color=OK, ha="center",
        fontweight="bold")
foot(ax, ["Q4 in plane stress, heat, Poisson",
          "the ordinary case --- nothing to think about"], OK)

# ---------- 2 : continuous + non-conforming ----------
ax = axes[0, 1]
head(ax, 2, "CONTINUOUS + NON-CONFORMING", BAD,
     "no gap, but the equation wants more than that")
# two straight element spans meeting at s=28 with clearly different slopes,
# against the smooth curve through the same three points
s1 = np.linspace(0, 28, 2); s2 = np.linspace(28, 56, 2)
ax.plot(22+s1, 38 - (5/28)*s1, color=INK, lw=3.4, zorder=5)
ax.plot(22+s2, 33 - (12/28)*(s2-28), color=INK, lw=3.4, zorder=5)
sm = np.linspace(0, 56, 160)
ax.plot(22+sm, -0.0044643*sm**2 - 0.053571*sm + 38, "--", color=GREY,
        lw=1.5, alpha=0.8, zorder=4)
ax.plot([50], [33], "o", ms=8.5, color=INK, zorder=7)
ax.annotate("slope jumps here", (50, 33), textcoords="offset points",
            xytext=(-26, -24), fontsize=11.8, color=BAD, fontweight="bold",
            ha="right", arrowprops=dict(arrowstyle="->", color=BAD, lw=1.9))
ax.text(50, 43.5, "deflection joins --- slope does not",
        fontsize=12.0, color=BAD, ha="center", fontweight="bold")
foot(ax, ["plate / beam elements: ACM (MZC), DKT",
          "a plate equation needs the slope to match too"], BAD)

# ---------- 3 : discontinuous + conforming ----------
ax = axes[1, 0]
head(ax, 3, "DISCONTINUOUS + CONFORMING", OK,
     "gap, but it sits on a boundary --- not inside a body")
quad(ax, [(20, 18), (46, 20), (46, 40), (20, 38)])
quad(ax, [(54, 20), (80, 18), (80, 38), (54, 40)])
ax.add_patch(Rectangle((46, 20), 8, 20, fc=OK, ec="none", alpha=0.17, zorder=2))
for yy in (24, 30, 36):
    ax.annotate("", (46.4, yy), xytext=(53.6, yy),
                arrowprops=dict(arrowstyle="<|-|>", color=OK, lw=1.9,
                                mutation_scale=10))
ax.text(50, 43.5, "two domains, not one", fontsize=12.0, color=OK,
        ha="center", fontweight="bold")
ax.text(33, 13.5, "body A", fontsize=11.0, color=GREY, ha="center")
ax.text(67, 13.5, "body B", fontsize=11.0, color=GREY, ha="center")
foot(ax, ["two bodies in contact;  a cohesive crack",
          "each piece is legal on its own domain"], OK)

# ---------- 4 : discontinuous + non-conforming ----------
ax = axes[1, 1]
head(ax, 4, "DISCONTINUOUS + NON-CONFORMING", AMB,
     "gap inside one body --- so who pays for it?")
quad(ax, [(20, 18), (46, 20), (46, 40), (20, 38)])
quad(ax, [(54, 20), (80, 18), (80, 38), (54, 40)])
ax.add_patch(Rectangle((46, 20), 8, 20, fc=AMB, ec="none", alpha=0.17, zorder=2))
for yy in (24, 30, 36):
    xs = np.linspace(46.3, 53.7, 40)
    ax.plot(xs, yy + 1.15*np.sin(np.linspace(0, 6*np.pi, 40)), color=OK,
            lw=1.7, zorder=6)
ax.text(50, 43.5, "one body, torn on purpose", fontsize=12.0, color=AMB,
        ha="center", fontweight="bold")
ax.text(50, 7.5, "penalty term present  $\\Rightarrow$  DG.  Works.",
        fontsize=11.8, color=OK, ha="center", fontweight="bold")
ax.text(50, 2.3, "nothing present  $\\Rightarrow$  Wilson Q6 on distorted quads,\n"
                 "Morley.  Fails the patch test.",
        fontsize=11.8, color=BAD, ha="center", va="center", fontweight="bold")

# ---------- axis headings ----------
fig.text(0.5, 0.965, "two questions, four boxes",
         fontsize=17.5, color=INK, ha="center", fontweight="bold")
fig.text(0.5, 0.925,
         "across:  is the gap legal for the equation you are solving?      "
         "down:  is there a gap at all?",
         fontsize=12.8, color=GREY, ha="center")
fig.text(0.030, 0.655, "NO GAP", fontsize=14.5, color=INK, rotation=90,
         ha="center", va="center", fontweight="bold")
fig.text(0.030, 0.235, "GAP", fontsize=14.5, color=INK, rotation=90,
         ha="center", va="center", fontweight="bold")
fig.savefig("q4_three_questions.pdf")
print("wrote q4_three_questions.pdf")
