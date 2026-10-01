# When does the Past Hypothesis select the subsystems?

### Quantum mereology as constrained tomography

Unreviewed draft posted for public feedback, October 2026.

This repository holds the manuscript, its LaTeX sources, and every script and raw
result behind the numbers in it.

---

## Status and provenance

Reproduced from the manuscript's own status box:

> **Not peer reviewed.** The results, proofs, code and text were produced by an AI
> system, Claude (Anthropic), in an extended conversation with a human user, who
> chose the direction and requested checks but did not derive results. No human
> expert has checked the proofs. Novelty is unverified: during this work the model
> twice claimed novelty that literature searches then refuted (Stoica's Remark 11;
> the Kech–Wolf theorem), and the use of the Past Hypothesis as a selection
> principle was first proposed by Stoica (Sec. VII.B of his 2022 paper).

Read it in that spirit. Appendix B of the extended version records every
computational check performed **and every error found**; `verification/dbg.py`
documents two failures in the test code itself.

---

## The argument in one paragraph

In Hilbert-space fundamentalism the world is a Hamiltonian $H$ and a state $\psi$,
and the split into subsystems has to be derived. Stoica showed no such derivation
is unique: every tensor product structure (TPS) making $H$ local is one of a
continuum related by unitaries commuting with $H$. This paper imposes a
strengthened Past Hypothesis as a selection rule — choose the division in which
some moment of the history is *simple* (near a product state, or near a matrix
product state) — and shows this turns the preferred-structure problem into
**constrained quantum state tomography**: recovering that simple moment (the
*bounce*) from a single projective measurement in the energy eigenbasis, with
simplicity as prior. The selection comes out exact for generic and random
eigenbases, and conjectural for local Hamiltonians.

---

## Layout

Two versions of the paper live here. The repository root holds the **short
version**; `extended/` holds the **extended version**, which adds the full
verification appendix.

```
.
├── main.tex                          short version
├── main_reddit.tex                   short version, variant for public posting
├── past_hypothesis_paper_reddit.pdf  built short version
├── fig1.pdf, fig2.pdf                figures
│
├── extended/
│   ├── main.tex                      extended version
│   ├── main_reddit.tex               extended version, posting variant
│   ├── fig1.pdf, fig2.pdf            copies, so this folder builds standalone
│   └── past_hypothesis_paper_extended.pdf
│
├── past_hypothesis_raw_results.tar.gz   18 .jsonl files behind Tables 1–2
│
├── eth_check.py                      ETH / random-amplitude separation check
├── mereology_toy_model.py            toy model: claims A–E on the TPS torus
├── mps_rule_scan.py                  the rule with MPS simplicity
├── past_hypothesis_scan.py           the rule with product-state simplicity
├── random_matrix_prediction.py       closed-form separation ratio Hel/D
├── rival_floor.py                    lowest entropy of any off-history reading
├── separation_constant.py            the constant kappa
│
├── verification/                     the Appendix B checks (see its README.txt)
├── option3/                          adversarial worst-case numerics, Table 7
├── time_reversal/                    time-reversal-invariant case
└── counterexample_search/            searches for counterexamples to the rule
```

Each top-level script carries a docstring with its exact usage line. Most take
positional arguments (`<n qubits> <seeds> <seconds per point>`) and print JSON.

---

## Building the paper

Each version builds from its own directory, since `\includegraphics` paths are
relative and there is no `\graphicspath`:

```bash
pdflatex main.tex && pdflatex main.tex          # short version, from repo root
cd extended && pdflatex main.tex && pdflatex main.tex   # extended version
```

> The figures in `extended/` are copies of the two at the root. The original
> source archive shipped `extended/main.tex` without them, so that version did
> not build standalone; the copies are the only change made to the sources here.

---

## Reproducing the numbers

Requires Python 3 with `numpy` and `scipy`.

**Verification suite** — run from `verification/`:

```bash
python3 v_part1.py   # analytic lemmas, constants of Lemmas A1–A4
python3 v_part2.py   # theorem constants, thresholds, MPS dimension
python3 v_part3.py   # chaos check, Tables 1–2 — needs the raw data, see below
python3 v_part4.py   # robust-lemma identity, ETH-A inequalities
```

`verification_log.txt` holds the recorded outputs to compare against.

**Raw data** — `v_part3.py` reads the `.jsonl` files unpacked beside it:

```bash
tar -xzf ../past_hypothesis_raw_results.tar.gz -C verification/
```

The archive holds 18 files: `ph_n{3,4,5}`, `ph2_ground_n{3..8}`,
`ph2_perturbed_n{4..8}`, and `mps_n{5..8}_chi2`. They are gitignored once
unpacked, so the tarball stays the single source.

---

## Known gaps

- **`option3/adv12b.py` cannot run as-is.** It reads a saved 268 MB eigenvector
  file that is too large to distribute. To use it, first run an `adv.py`-style
  diagonalisation to regenerate that file. The 12-qubit results it produced are
  recorded in `option3/README.txt`: 0.92, 0.87 (seed 101) and 0.89, 0.93
  (seed 202), with the raw output in `multiseed.log` and `cold.log`.
- **The uniform separation the rule needs is open.** It is proved for generic and
  Haar-random eigenbases, and supported for local Hamiltonians only by simulation
  (chaotic chains up to 12 qubits) plus an ETH-type hypothesis giving pairwise
  separation in expectation.
- **No human expert has checked the proofs.** See the status box above.

---

## Feedback

Issues and corrections are welcome — that is what this repository is for. If you
find an error in a proof or a number, please open an issue.
