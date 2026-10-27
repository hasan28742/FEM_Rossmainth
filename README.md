# FEM Fall 2027 — lecture notes

Typeset notes for a graduate finite element course (math department), annotated
for a mechanical engineering reader.

## Layout

```
main.tex            master document; \input one file per lecture
preamble.tex        packages, colours, the annotation boxes
macros.tex          FEM notation shortcuts (\Honez, \bl{u}{v}, \Kmat, ...)
lectures/lecNN.tex  one file per lecture
scans/              raw OneNote PDF exports go here
scans/png/          per-page images generated from those PDFs
figures/            TikZ/PDF figures
build/              compiler output (build/main.pdf)
```

## Documents

| Source | Output | What it is |
|---|---|---|
| `main.tex` | `build/main.pdf` | annotated notes: transcription + commentary boxes |
| `lectures.tex` | `build/lectures.pdf` | **the transcription alone** — same sources, commentary suppressed |
| `kink_vs_jump.tex` | `build/kink_vs_jump.pdf` | standalone note: kink vs jump, the delta, and what it decides in FEA |

Both share `preamble.tex` and `macros.tex`, so the boxes and notation stay
consistent.

## Build

```bash
./build.sh                 # everything
./build.sh main            # annotated notes
./build.sh lectures        # clean transcript, no commentary
./build.sh kink_vs_jump    # the standalone note
```

`lectures.tex` re-uses `lectures/lec*.tex` unchanged — it just sets
`\transcripttrue` and redefines `roadmap`/`intuition`/`mechanics`/`pitfall`/
`dictionary` and `\todoq` to print nothing. So there is one copy of each lecture, and the two
PDFs can never drift apart. `recall` is kept but rendered without its box,
because it carries lecture material (the delta properties) and the equation
labels the main text cites.

`latexmk` is installed but unusable on this machine: the system Perl is missing
`File::Copy`. `build.sh` calls `pdflatex` directly instead.

TeX here is TinyTeX, which ships almost nothing by default. If a build fails
with `File 'x.sty' not found`:

```bash
tlmgr search --global --file "/x.sty"    # find the package name
tlmgr install <package>
```

## Getting notes out of OneNote

1. In OneNote: **File → Export → Section → PDF**. Export one section per
   lecture if you can; it keeps the transcription units small.
2. Drop the PDF in `scans/`.
3. Convert to page images:

   ```bash
   ./tools/scans2png.sh scans/lecture01.pdf
   ```

   Writes `scans/png/lecture01-01.png`, `-02.png`, ... at 200 dpi.
4. Those PNGs are what gets read and transcribed into `lectures/lecNN.tex`.

Handwriting transcription is not OCR here — the pages are read directly, so
diagrams, marginalia and arrows between equations survive as structure rather
than being dropped.

## The annotation boxes

The mathematical content is transcribed as given. Everything in a coloured box
is added commentary, kept visually separate so it is never confused with what
the lecturer actually said.

| Environment | Purpose |
|---|---|
| `roadmap` | the bullet-point big picture, at the top of every lecture |
| `intuition` | what an object *is*, stripped of formalism |
| `mechanics` | the same statement as a bar / beam / slab / stiffness matrix |
| `dictionary` | two-column math ↔ mechanics translation table |
| `pitfall` | where the two vocabularies describe one thing and sound unrelated |
| `recall` | prerequisite the lecturer assumed |
| `\todoq{...}` | something illegible or not followed — office hours question |

## Lectures transcribed so far

| File | Content | Source |
|---|---|---|
| `lectures/lec01.tex` | Why FEM; the two-point BVP; self-adjointness; the space `V` | `1st lecture.pdf` pp. 1–6 |
| `lectures/lec02.tex` | Weak solutions; the Heaviside example; Claims 1–3; strong vs weak | `1st lecture.pdf` pp. 7–15 |
| `lectures/lec03.tex` | Energy minimisation; min ⇔ variational; CG(1) and hat functions | `3rrd.pdf` pp. 16–22 |
| `lectures/lec04.tex` | The discrete system: stiffness matrix and load vector | `4th.pdf` |
| `lectures/lec05.tex` | Nonhomogeneous Dirichlet BCs (half-hats, lifting); Neumann BCs | `5th.pdf` |
| `lectures/lec06.tex` | Neumann assembly; Galerkin orthogonality; first a priori estimate | `6th.pdf` |
| `lectures/lec07.tex` | A priori error, energy norm; the interpolant `Pi_h u` | `7th.pdf` |
| `lectures/lec08.tex` | A priori error in `L2`; stability constant; the dual problem | `8th.pdf` |
| `lectures/lec09.tex` | A posteriori error; the adaptive algorithm | `9th.pdf` |
| `lectures/lec10.tex` | **Lectures 10+11 combined:** CG(2); the beam equation; CG(3) basis | `missing.pdf` pp. 1-6 |
| `lectures/lec12.tex` | CG(3) assembly: the Hermite stiffness matrix and load vector | `12th.pdf` (cross-check `missing.pdf` pp. 7-9) |

**`missing.pdf`** is a classmate's CamScanner PDF (no text layer) covering the
lectures absent from my own notebook. It actually spans **three** lectures, not
two: pp. 1-3 are Lecture 10 (CG(2)), pp. 4-6 are Lecture 11 (beam equation,
energy minimisation, the CG(3) Hermite basis), and pp. 7-9 repeat the same
material as my own `12th.pdf` (the CG(3) assembly). The two sources agree, so
Lecture 12 is transcribed from `12th.pdf` and `missing.pdf` is used as the
cross-check there.

**Cross-check source:** `scans/peer_notes.pdf` — another student's handwritten
notes (much cleaner), covering lectures 1–6. Useful for resolving ambiguous
symbols. It confirms the course is **MATH 6660, Prof. James Rossmanith**.

**Watch the OneNote exports** — they export the *whole section*, not one
lecture, so each new PDF re-contains everything before it. `3rrd.pdf` is 22
pages of which only 16–22 are new. Check where the new material starts before
transcribing, and delete the duplicate page images. (The duplicated pages are
not byte-identical between exports — the page footer carries the section name,
which changes — so `cmp` will not detect them. Compare visually.)

## Source notes

OneDrive is mounted at `~/onedrive` via rclone (systemd user unit
`onedrive-mount.service`). It is a live view of the cloud, not a synced copy.
Course material lives under `~/onedrive/Course/Fall_2026/`,
`~/onedrive/FEM_Notes/` and `~/onedrive/FEM_Learning_Hasan/`.
