# EgoScore — Validation Report

**EgoVerse Data Optimization & Evaluation Suite, Track 1.** 2026-08-15.

This is the detailed version. For the short one, see the [README](../README.md).

---

## What we set out to test

> At a fixed number of clips, does a filtered, deliberately varied subset train a better
> model than the same number of clips picked at random?

**Result: yes, by a small but consistent margin**, with one important caveat about what
that does and does not mean. Both are set out below.

## Results

| Selection method | Comparisons won | Prediction error vs random | Wilcoxon p |
|---|---|---|---|
| Log-determinant | 40/40 | **3.53% lower** | 1.8e-12 |
| k-center | 40/40 | **3.45% lower** | 1.8e-12 |
| Facility location | 33/40 | **2.14% lower** | 1.6e-08 |
| Random, without the filtering step | 24/40 | no measurable difference | 5.2e-01 |
| Deliberately narrow *(designed to fail)* | 1/40 | **5.83% higher** | 5.5e-12 |

Every method was compared against random selection using the same budget, the same
held-out clips, and the same model settings. 10 repeats x 2 budget sizes x 2
held-out conditions = **40 paired comparisons** per method. Significance from a
two-sided Wilcoxon signed-rank test on the paired differences.

- **Varied selection wins consistently.** Log-determinant wins 40/40 comparisons at 3.53% lower error (p=2e-12); k-center 40/40; facility location 33/40.
- **The filtering step makes no measurable difference here.** Skipping it scores 24/40 at +0.02%, sign-test p=0.27 — indistinguishable from no effect. That is expected: filtering removes only 49 of 572 clips in this collection, so there is very little for it to take out. We measured the filtering step separately, on a collection where it removes 1 in 7 — see the section below.
- **The control designed to fail does fail.** A selection drawn almost entirely from a handful of people and rooms wins only 1/40 at +5.83% (p=5e-12). The next section explains why that matters more than the headline number.

## Why the control matters more than the headline

A result showing only *our method beat random by 3%* is hard to trust. The effect is small,
the score is a proxy, and the pool is a few hundred clips. The obvious failure mode is a
measurement too noisy to distinguish anything, in which case a 3% difference is chance.

So we included a selection that *should* lose, for a reason established independently by the
EgoVerse authors: variety in demonstrators and scenes improves generalisation. This control
uses the identical budget but draws almost entirely from a handful of people and rooms.

At K=25% it is **+10.9%** on unseen operators and **+7.4%** on unseen scenes — a large, unambiguous, correctly-signed effect.

So the measurement *can* detect an effect of the kind the paper reported. The smaller margins
above are therefore measurements rather than noise.

The control also weakens at the larger budget (+3.0% / +2.0%), which is what should happen: with half the pool you cannot concentrate the selection very much, so it converges toward random. A control that stayed constant would have been grounds for suspicion.

## What this does not show

**Training on the full collection remains the best option.** It beats every 25% and 50%
subset we selected, including the strongest one:

| | unseen operator | unseen scene |
|---|---|---|
| random, K=25% | 0.11527 | 0.10221 |
| best selector (dpp), K=25% | 0.10893 | 0.09810 |
| **all gated data (100%)** | **0.10226** | **0.09209** |

The claim is therefore not "discard 75% of the data at no cost". It is:

> At a quarter of the budget, diversity-aware selection closes **49%** (unseen operator) / **41%** (unseen scene) of the gap between a random quarter and using everything.

That is the figure that matters when the full collection is not an option — deciding what to
record next, what to pay for annotation, or what will fit on a robot.

## Method

**Data.** Clothes-folding clips from lab `rl2` — 572 clips, 20 people, 16 rooms. This is the
only collection in EgoVerse that records both who filmed each clip and where. `microagi`
(9,896 clothes-folding clips) records neither; `mecka` has 2 rooms. Without both, neither
held-out condition can be constructed.

**Score.** The model is shown 10 frames of where both hands have been and predicts their
positions over the next 30 frames (one second). We score squared error against what the
person actually did. Predictions are made *relative to the current hand position*, so the
model cannot score well by memorising which room it is in. The model is closed-form ridge
regression, so it has no training randomness: any difference between conditions comes from
the data rather than from optimiser noise.

**Held-out conditions.** For each repeat we hold back 25% of the people and 25% of the rooms.
The unseen-person evaluation excludes held-back rooms and vice versa, so the two conditions
measure different things. Training excludes both.

**Statistics.** Absolute error varies substantially between repeats, because each repeat holds
back a different set of people and rooms, shifting every method's score at once. Reporting raw
means with error bars across repeats would bury the effect in that variation. Since all methods
within a repeat see an identical split and identical evaluation clips, we report **paired**
differences and count wins.

## The filtering step

49 of 572 clips removed (8.6%). Every rule is arithmetic on the
recorded hand and head positions; no learned model is involved, and each removed clip records
which rule caught it.

| rule               | signal           | test                  |   n_flagged |      pct | meaning                                                                               |
|:-------------------|:-----------------|:----------------------|------------:|---------:|:--------------------------------------------------------------------------------------|
| tracking_dropout   | nan_max          | gt 0.05               |           0 | 0        | non-finite pose/keypoint values in >5% of frames                                      |
| frozen_tracker     | frozen_run_max_s | gt 2.0                |           0 | 0        | pose held bit-identical for >2s (tracker dropout)                                     |
| too_short          | duration_s       | lt 3.0                |           0 | 0        | shorter than 3s — cannot contain a fold                                               |
| no_motion          | path_len_total   | lt 0.1                |           0 | 0        | hands travelled <10cm in total                                                        |
| jittery_hand_track | hand_jitter_frac | gt 0.01               |           6 | 1.04895  | over 1% of frames have a hand moving >10 m/s — the tracker is jumping, not the person |
| head_pose_jump     | head_speed_max   | gt 20.0               |           7 | 1.22378  | head jumps >20 m/s in one frame — SLAM relocalised mid-episode                        |
| hands_far_off_axis | offaxis_max      | gt 0.5 (labs: rl2)    |          37 | 6.46853  | a hand is held >45 deg off the camera axis for most of the episode                    |
| runaway_length     | duration_s       | gt 3.0x median (278s) |           0 | 0        | more than 3x the median episode length — an un-segmented recording, not one demo      |
| suspiciously_short | duration_s       | lt 0.15x median (14s) |           1 | 0.174825 | under 15% of the median episode length — a fragment, not a demo                       |
| ANY (dropped)      | -                | -                     |          49 | 8.56643  | union of all rules over 572 episodes                                                  |

**What this says about the data:** this collection is largely clean. No clip has missing or
frozen tracking. Almost every removal is for hand placement. A filter designed around imagined
failure modes would find nothing here — the interesting prevalence is in other collections, in
the cross-collection section below.

### A rule we got wrong

An earlier version had a seventh rule that projected hand keypoints into the camera image
and flagged episodes where the hands left frame. It dropped 23 episodes. We then rendered
those frames for a figure and the hands were plainly visible in all of them.

We tried two projection models — keypoints as camera-frame points, and keypoints
transformed by the head pose first. Under both, **zero of the 42 keypoints** land inside
the image on frames where the hands are obviously visible
(`scripts/22_projection_check.py`). We could not establish the stored convention, so we
removed the rule rather than ship a third guess. Every number it produced was unfounded.

The six surviving rules use durations, distances and NaN counts only. None touches camera
geometry, which is also why they compare cleanly across labs.

The histogram of the deleted signal looked entirely reasonable. Rendering the frames is
what caught it, and we only did that because we wanted a picture for a slide.

One trap worth recording: zarr arrays are zero-padded up to a chunk boundary, and the true
length is `total_frames` in the group attrs. Reading the raw array without truncating
produces a run of identical trailing frames that looks exactly like a frozen tracker. Our
frozen-tracker count is zero *because* we truncate; without that step it would have been
near 100% and the gate would have thrown away the entire dataset.

## Sensitivity: did we tune the proxy to get this?

The obvious attack is that we picked a ridge configuration that happened to flatter our
selector. So we re-ran the K=25% comparison across a grid of proxy configurations —
regularisation strength (0.1 / 1 / 10), feature-map width (256 / 512 / 1024), history
length (5 / 10 / 20 frames), and prediction horizon (15 / 30 / 60 steps) — re-tuning
nothing per condition. If the conclusion only held at one setting, it would not be a
conclusion.

| Selector | Configurations where it beats random | Mean | Worst case |
|---|---|---|---|
| `dpp` | 9/9 | -4.11% | -3.68% |
| `kcenter` | 9/9 | -4.42% | -3.63% |
| `curated` | 9/9 | -2.66% | -1.65% |
| `degenerate` | 0/9 | +11.64% | +16.89% |

The ranking is unchanged in every configuration tested, and the positive control loses in
every configuration. Effect sizes here are larger than the headline because this sweep
uses K=25% only, where selection has the most room to matter; the headline averages 25%
and 50%.

Full grid: [`reports/sensitivity_summary.csv`](sensitivity_summary.csv).

## Cross-lab audit: does the gate discriminate?

A gate that fires at the same rate everywhere is measuring its own thresholds, not
quality. We ran it against a 400-episode sample of `mecka` fold_clothes as a check.

Only the intrinsics-independent signals are compared: hands-out-of-frame depends on
camera intrinsics that we verified for the Aria rig used by `rl2` and have not verified
for `mecka`, so reporting it cross-lab would be a number we cannot stand behind.

| lab      |   n_episodes | tracking_dropout   | frozen_tracker   | too_short   | no_motion   | runaway_length   | suspiciously_short   | ANY   |   median_dur_s |   median_motion |
|:---------|-------------:|:-------------------|:-----------------|:------------|:------------|:-----------------|:---------------------|:------|---------------:|----------------:|
| mecka    |          400 | 0.0%               | 0.0%             | 1.2%        | 0.0%        | 0.0%             | 0.0%                 | 1.2%  |              7 |            0.34 |
| microagi |         1200 | 0.0%               | 0.0%             | 0.0%        | 0.0%        | 13.9%            | 0.0%                 | 13.9% |             11 |            0.38 |
| rl2      |          572 | 0.0%               | 0.0%             | 0.0%        | 0.0%        | 0.0%             | 0.2%                 | 0.2%  |             93 |            0.55 |

**Both labs are clean on tracking dropout and frozen trackers — 0.0% in each.** That is a
genuine finding about EgoVerse rather than a null result: the pose pipelines are solid,
and a curation engine built around imagined tracking failures would find nothing to do.

**The more consequential finding is that an "episode" is not a common unit across labs.**
Median episode duration is 93 s in `rl2` and 6.6 s in `mecka` — a factor of ~14. `rl2`
episodes are long multi-fold sessions; `mecka` episodes are short single-action clips.
Anyone budgeting curation in episode counts across labs is comparing incommensurable
units, and a subset of "1,000 episodes" means wildly different things depending on where
they came from. Budgeting in frames or seconds would be the safer default.

## Limitations

Stated plainly, because these are the first things worth attacking.

1. **Avg-MSE is a proxy, not robot success.** The EgoVerse authors use it while saying so:
   *"this metric does not directly measure downstream robot performance [but] provides a
   stable signal for comparing generalization."* We adopt it on the same terms. It ranks
   subsets; it does not predict success rate.
2. **The proxy policy is proprioceptive, not visual.** Images for this slice are 84.6 GB
   against 2.3 GB for pose arrays. That was a deliberate trade: a curation engine that
   costs more than the training it saves is not worth running. But it means the proxy
   cannot be task-conditioned on what the scene looks like, and a visual policy might rank
   subsets differently.
3. **One task, one lab.** We cannot claim the method transfers across tasks. The slice was
   forced by metadata availability, not chosen for favourability.
4. **Small effect, small pool.** ~3% Avg-MSE on a few hundred episodes. The positive
   control is what makes this interpretable rather than decorative.
5. **Facility location was our a priori pick and it is not the winner.** `dpp` and
   `kcenter` both beat it. We are reporting that rather than quietly promoting the winner
   to headline method: on this slice, *spread* appears to matter slightly more than
   *coverage*. With 10 seeds `dpp` is cleanly ahead of `curated`, but `dpp` and
   `kcenter` are within noise of each other.

## Reproducing

```bash
bash scripts/run_all.sh
```

## Figures

![conditions](figs/conditions.png)
![paired](figs/paired.png)
![coverage](figs/coverage.png)
![gate](figs/gate.png)
