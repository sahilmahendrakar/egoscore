# EgoScore — A Curation Engine for EgoVerse

**EgoVerse Data Optimization & Evaluation Suite — Track 1: The Curation Engine**

> *Which episodes are worth training on?*

EgoVerse ships 1,362 hours / ~80k episodes of egocentric human demonstrations. Not
all of it helps. This repo picks a subset that does, and — the part that matters —
**measures whether the pick was actually better than picking at random.**

## The claim we are testing

> At a fixed episode budget K, a quality-gated, coverage-maximizing subset trains a
> better policy than K episodes sampled uniformly at random.

We test it with the EgoVerse paper's own offline proxy metric (Avg-MSE) and its own
two generalization axes (held-out **unseen demonstrators**, held-out **unseen scenes**).

## Status

🚧 Built during the one-day sprint, 2026-08-15. See [`PLAN.md`](./PLAN.md).

## Pipeline

```
episode table (Postgres)
        │
        ├─ quality gate ──────────► drop broken episodes (tracking dropout,
        │                            frozen pose, out-of-frame hands, degenerate
        │                            duration, near-zero motion)
        │
        ├─ episode embedding ─────► trajectory statistics + DINOv2 visual features
        │
        └─ submodular selection ──► facility-location coverage maximization
                                     (ablations: DPP log-det, k-center, random)
                                            │
                                            ▼
                              validation harness: train identical
                              BC head on each subset, compare Avg-MSE
```

## Honest limitations

Stated up front, because they are the first thing worth asking about:

- **Avg-MSE is a proxy, not robot success.** The EgoVerse authors say so themselves and
  use it anyway as a stable comparative signal. We adopt their protocol and report it
  as a proxy. See `PLAN.md` for why we still think it is the right call.
- **Small scale.** One flagship task. Curation effects are noisy at this scale, which is
  exactly why we run a positive control and report error bars over seeds.

## License

MIT
