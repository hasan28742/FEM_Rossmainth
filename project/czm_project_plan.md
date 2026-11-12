# MATH 6660 final project — CZM under compression (Camanho foundation)

Proposal due Mon Nov 9 2026 (text: `proposal.txt`, option D).
Talk ~10 min (Dec 7/9/11/16). Write-up ≤ 5 pages, due Wed Dec 16 12:00.
Code source: `mowry-cdm-czm/` (private research repo, cloned 2026-11-08; do not edit it —
run scratch copies).

## Story (one slide each)

1. **Foundation — Camanho's law** (TM-2002 eqs 27–46; J. Compos. Mater. 2003).
   Onset: quadratic criterion, "compressive normal tractions do not affect delamination
   onset" (eq 27). Onset jump δ_m⁰ (33), B-K failure jump δ_mᶠ (40), damage d (44),
   constitutive D_sr (43) — in compression K is restored on the normal component
   (no interpenetration), no friction.
2. **FE formulation.** Mesh split along the fracture (BreakMeshByBlock), displacement
   jump [[u]] = u⁺ − u⁻, weak form gains ∫_Γ t([[u]]) · [[v]] dΓ. Penalty K = stiffness
   of the uncracked interface. Kernel path (`BiLinearMixedModeTraction`) vs mortar
   user-object path (`BilinearMixedModeCohesiveZoneModel`, used in the sweep).
3. **Evolution in my framework (manuscript §2.2–2.3, §4).** Camanho law + mortar
   penalty contact for t_N < 0 + Coulomb friction ‖t_t‖ ≤ μ|t_N|, cohesive and friction
   tangential tractions *added*.
4. **Single-element evidence (run on stock conda MOOSE, 2026-11-08):**

   | case | result |
   |---|---|
   | Mode I opening (G_Ic = 0.2) | dissipated 0.201 J/m² (ratio 1.006) |
   | Mode II, closed interface (G_IIc = 0.14 in input) | dissipated 0.198 J/m², δ_f = 40.0 µm (ratio **1.414 = √2**) |
   | Mode I compression, kernel and mortar | run, no tensile traction |
   | friction utilisation (compress, then shear) | runs, 40 steps, 3.8 s |
   | inclined 10° no-penetration | runs, 1.3 s |

   Kernel decks need `interface_name = interface` + `split_interface = false` in
   `BreakMeshByBlockGenerator` on the conda MOOSE (older default names the side set
   `lower_block_upper_block`). Scratch copies:
   `/tmp/claude-1000/-home-hasan-Documents-FEM-Fall2027/7016c99a-de4a-4f6d-9828-ea9c73aae6d8/scratchpad/czm_single/`.
5. **What compression exposes.**
   - Onset and toughness are pressure-independent (eq 27) — for a fracture under
     23.4 MPa confinement this is the main physical gap.
   - **√2 on the closed branch.** Eq (40) writes δ_mᶠ = √((δ₁ᶠ)² + (δ₂ᶠ)²) for δ₃ ≤ 0;
     the text below it says δ_mᶠ = δ_shearᶠ in shear mode. MOOSE (both paths) reads it
     literally: δ_f = √2·2G_IIc/S while δ₀ = S/K, so a closed interface dissipates
     √2·G_IIc; a slightly open one (B-K, β→∞) gives 2G_IIc/S → 41 % energy jump at δ_N = 0.
     Code: `BiLinearMixedModeTraction.C:183`, `BilinearMixedModeCohesiveZoneModel.C:232`.
   - Friction is added at full strength, not on the damaged fraction.
6. **Future work — how the CZM should evolve in compression.**
   - Friction on the damaged area only: Alfano & Sacco 2006.
   - Compression-enhanced mode II toughness (measured, modelled with cohesive + friction):
     Catalanotti, …, Camanho 2017 — Camanho's own group extending his law.
   - Fix the closed branch to δ_shearᶠ = 2G_IIc/S (continuous at δ_N = 0).

## Course hooks (math grader)

Weak form with a discontinuous trial space; penalty K ↔ conditioning; softening branch
→ tangent loses positive definiteness → Lax–Milgram (Lecture 13) no longer applies →
why viscous regularisation is used.

## References (verified)

- Camanho, Dávila, de Moura (2003) J. Compos. Mater. 37(16):1415–1438, doi:10.1177/0021998303034505
- Camanho, Dávila (2002) NASA/TM-2002-211737 — https://ntrs.nasa.gov/citations/20020053651
- Alfano, Sacco (2006) IJNME 68(5):542–582, doi:10.1002/nme.1728
- Catalanotti, Furtado, Scalici, Pitarresi, van der Meer, Camanho (2017) Compos. Struct. 182:153–163, doi:10.1016/j.compstruct.2017.09.014
- Snozzi, Molinari (2013) IJNME 93(5):510–526, doi:10.1002/nme.4398 (optional: cohesive + Coulomb contact)

## Flag for the manuscript (not a course item)

Appendix A caption says G_IIc = 0.2 J/m², δ_f = 2G_c/N = 40 µm; the Mode II input uses
G_IIc = 0.14 and matches only because the plot script uses MOOSE's √2 formula. In the
sweep the interface is closed, so its effective mode II energy is √2 × the tabulated G_IIc.
