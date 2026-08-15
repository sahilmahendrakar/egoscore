# EgoScore — Build Plan

**Track 1: The Curation Engine.** One-day sprint, 2026-08-15. Submission 4:45pm.

---

## 0. The thesis

Dumping everything into training is inefficient and can hurt performance. So:

> **At a fixed episode budget K, a quality-gated, coverage-maximizing subset trains a
> better policy than K episodes sampled uniformly at random.**

Everything in this repo exists to state that claim precisely and then test it.

The brief asks for *"keep/drop recommendations plus a validation report using proxy
metrics."* We deliver both, and we pick the proxy metric the EgoVerse authors
themselves used so the choice is not ours to defend alone.

---

## 1. Data scope

**One flagship task, all labs** — `fold_clothes` across `rl2`, `eth`, `song`, `wang`.

Rationale:
- A homogeneous task makes trajectory-space coverage a meaningful notion. Across
  wildly different tasks, "coverage" degenerates into "task balance," which is a
  much less interesting claim.
- Multiple labs give real distribution shift (different objects, scenes, demonstrators)
  which is what the held-out axes need in order to be non-trivial.
- It bounds the download. Flagship EgoVerse-A is ~2,385 episodes across 6 tasks total,
  so this slice is small enough to pull and embed inside the clock.

If the slice turns out too small for a stable experiment (< ~300 episodes), fall back to
adding the Mecka `fold_clothes` episodes from EgoVerse-I, which is a much larger pool.

---

## 2. Pipeline

### 2.1 Quality gate — `egoscore/quality.py`

Deterministic, per-episode, no learned components. Every rule is inspectable and every
dropped episode carries a reason string.

| Signal | Failure it catches |
|---|---|
| Fraction of frames with NaN / missing hand keypoints | tracking dropout |
| Longest run of byte-identical consecutive poses | frozen tracker |
| Fraction of frames with keypoints projecting outside the image | hands out of frame |
| Episode duration z-score within task | truncated / mis-segmented episodes |
| Total motion energy (integrated keypoint velocity) | idle or near-empty episodes |
| Final-frame vs first-frame pose displacement | episodes that never went anywhere |

Output: `reports/quality.csv` — one row per episode, every signal, plus `keep`/`drop`
and `drop_reason`.

**This is where the Track 3 idea survives.** A drop-detector is the front half of a
curation engine; it just is not the whole submission.

### 2.2 Episode embedding — `egoscore/embed.py`

Each episode → one fixed-length vector, concatenating:

- **Trajectory features (CPU, always available):** per-hand EE-pose statistics in the
  head frame — path length, velocity/acceleration moments, workspace bounding volume,
  bimanual coordination (correlation between hands), approach/retreat structure, duration.
- **Visual features (Modal GPU, stretch):** DINOv2 embeddings of N sampled frames per
  episode, mean-pooled, plus the first- and last-frame embeddings kept separately
  (start state and end state carry different information).

Both are L2-normalized and whitened before concatenation so neither block dominates.

**Fallback:** if Modal setup runs long, trajectory features alone are sufficient for the
full experiment. The visual block is strictly additive and is gated behind a flag.

### 2.3 Selection — `egoscore/select.py`

Given the embedding matrix and budget K:

- **`facility_location`** *(our method)* — greedy submodular maximization of coverage.
  Monotone submodular, so greedy carries a (1 − 1/e) approximation guarantee. That
  guarantee is a real answer to "why this algorithm."
- **`dpp_logdet`** — determinantal point process log-determinant, a diversity ablation.
- **`kcenter`** — classic k-center greedy, a coverage ablation.
- **`random`** — the baseline that actually matters.
- **`degenerate`** — *positive control*: restrict to a single demonstrator in a single
  scene. See §3.

All selectors run **after** the quality gate, so we are measuring the value of
*selection*, not of *filtering*. Filtering value is measured separately by running
`random` with and without the gate.

---

## 3. Validation — `egoscore/validate.py`

This is the half of the project that makes the other half defensible.

### 3.1 The proxy metric

**Offline Avg-MSE** on held-out episodes: train a policy to predict an action chunk,
report mean squared error against the recorded human action chunk.

The EgoVerse paper: *"we report primarily the offline Avg-MSE metric... While this
metric does not directly measure downstream robot performance, it provides a stable
signal for comparing generalization."*

We adopt it for exactly that reason and **report it as a proxy, never as success**.

### 3.2 The model

Frozen DINOv2 features → small MLP action head predicting a 30-step action chunk of
EE poses. Frozen backbone means each training run is minutes, which is what lets us
afford seeds and conditions instead of one hero run.

Identical architecture, identical hyperparameters, identical step count across every
condition. Only the *training subset* changes. Anything else would confound the result.

### 3.3 The held-out axes

Two splits, mirroring the paper's controlled-diversity experiments:

- **Unseen demonstrators** — held-out `operator` values, same scenes.
- **Unseen scenes** — held-out `scene` values.

A curation method that helps on both is generalizing. One that helps on only one is
telling us something more specific, and we report that rather than averaging it away.

### 3.4 Conditions

| Condition | Purpose |
|---|---|
| `all-N` | ceiling: everything that passes the gate |
| `random-K` | **the baseline to beat** |
| `curated-K` | facility location |
| `dpp-K`, `kcenter-K` | selection ablations |
| `degenerate-K` | **positive control** |
| `random-K-nogate` | isolates the value of the quality gate |

× 3 seeds, at K ∈ {25%, 50%} of the gated pool.

### 3.5 The positive control, and why it is load-bearing

**Curated-vs-random is genuinely at risk of being a null result** at this scale. One
shot, noisy metric, few hundred episodes. Pretending otherwise would be the weakest
thing in the submission.

So we include `degenerate-K`: same budget K, but all episodes drawn from one
demonstrator in one scene. The paper already established that demonstrator and scene
diversity improve generalization — this is a **known-true effect we are replicating**.

- If `degenerate-K` is clearly worse → the harness is **provably sensitive**. Any null
  in curated-vs-random is then a real measurement ("curation does not pay at this
  scale"), not a broken pipeline.
- If `degenerate-K` is *not* worse → our harness cannot detect an effect the paper
  found, and we say so plainly. That is a negative result about our setup, and
  reporting it is more defensible than quietly shipping a chart.

Either branch produces an honest, presentable finding. That is the point.

---

## 4. Deliverables

1. **`reports/keep_drop.csv`** — per-episode keep/drop with reason and coverage score.
2. **`reports/validation.md`** — the report: conditions, seeds, error bars, the plots,
   and an explicit limitations section.
3. **Dashboard** — plots of the coverage-vs-budget curve and Avg-MSE by condition, with
   per-episode drop reasons and thumbnails so the drop decisions are inspectable.
4. **One summary slide** — the headline number, the method in one diagram, the caveat.
5. **This repo**, runnable end to end via `scripts/run_all.sh`.

---

## 5. Schedule and cut order

| Time | Milestone |
|---|---|
| 12:30 | Data access resolved, `fold_clothes` slice pulling |
| 13:15 | Quality gate running, `quality.csv` produced |
| 14:00 | Trajectory embeddings + selection working; **decide on Modal/visual** |
| 14:45 | Validation harness green: one full condition trained end to end |
| 15:30 | All conditions × seeds swept |
| 16:00 | Report, plots, dashboard |
| 16:30 | Slide, README, final commit |
| **16:45** | **Submit** |

**Cut order when behind** (first to go at the top):

1. Dashboard → static matplotlib plots
2. Visual/DINOv2 features → trajectory features only
3. `dpp` and `kcenter` ablations → keep `curated` / `random` / `degenerate` only
4. K sweep → single K = 50%
5. 3 seeds → 2 seeds

**Never cut:** the positive control, the error bars, the limitations section, and a
repo that runs. "Does it run" is judging criterion #1 and a working narrow result beats
a broad broken one.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| **Data access blocked / RDS unreachable** | Fallback: enumerate episodes from the R2 bucket prefix and read metadata from each `zarr.json`, skipping SQL. Costs us `lab`/`operator`/`scene`, which weakens the held-out-demonstrator split specifically. |
| Null result on curated-vs-random | Positive control (§3.5) guarantees a presentable finding either way |
| Slice too small for stable stats | Fall back to adding Mecka `fold_clothes` from EgoVerse-I |
| Modal setup eats the clock | Visual features are behind a flag; trajectory-only path is the default |
| Training loop bugs eat the clock | Build the harness against random subsets *first*, before any selection logic exists |

---

## 7. What we are explicitly not doing

- **No LLM-as-judge anywhere.** Every scoring component is deterministic and inspectable.
- **No claim about robot success rates.** We measure an offline proxy and we label it as one.
- **No second track.** The brief says narrow and working, not broad and broken.
