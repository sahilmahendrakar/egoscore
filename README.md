# EgoScore

**Which training videos are worth keeping?**

EgoScore filters a library of robot-training videos, then selects a small subset of what
remains that is deliberately varied rather than random. On the collection we tested, a quarter
of the clips chosen this way trained a **3.5% more accurate** model than a quarter chosen at
random, winning all 40 head-to-head comparisons.

🧭 **[Interactive demo](https://claude.ai/code/artifact/0db5ec63-9657-4738-854a-2703290324cc)** ·
📊 **[Slides](https://claude.ai/code/artifact/f9b606c2-ad6d-488b-ab9f-727c9875e1f8)** ·
📄 **[Full write-up](reports/validation.md)**

Built in a day for the EgoVerse Data Optimization & Evaluation Suite, Track 1.

---

## The problem

[EgoVerse](https://github.com/GaTech-RL2/EgoVerse) is a library of videos of people performing
everyday tasks, recorded from a camera worn on the head. Robots learn manipulation from them.

It contains hundreds of thousands of clips. Some are unusable — the hand tracking fails, or the
recording covers a whole session rather than a single task. Many others are near-duplicates of
one another, which add training time without adding information.

There is currently no established way to decide which clips are worth training on. This is an
attempt to decide it with a measurement rather than a judgement call.

---

## How it works

### 1 · Remove clips that are unusable

Eight checks, computed directly from the recorded hand and head positions. No learned model is
involved, and each rejected clip records which check caught it.

| Check | Threshold |
|---|---|
| Position data missing or non-numeric | in more than 5% of frames |
| Position repeats identically | for more than 2 seconds |
| Hand jumps between frames | faster than 10 m/s — folding laundry, a hand moves under 2 m/s |
| Head jumps between frames | faster than 20 m/s, which is the head-tracking solver resetting |
| Clip too short to contain a task | under 3 seconds |
| Hands barely move | under 10 cm of total travel |
| Clip far longer than normal for its lab | over 3× that lab's median length |
| Hands held out to one side | more than 45° from the camera's centre line, for most of the clip |

On the collection we test against, this removes **49 of 572** clips. On the largest collection
in EgoVerse it removes **1 in 7**.

### 2 · Select a varied subset of the rest

Each clip is summarised as 41 numbers describing how the person moved: distance and speed of
each hand, the volume of space each hand swept through, how strongly the two hands moved
together, how far the fingers spread, and how much the head turned.

Two clips are treated as similar when those numbers are close. We then compare four ways of
choosing a subset of fixed size:

| Method | What it optimises |
|---|---|
| **Log-determinant** | the volume the chosen clips span, so no two selections are redundant |
| **k-center** | repeatedly takes the clip furthest from everything chosen so far |
| **Facility location** | ensures every clip in the library has a similar one in the chosen set |
| **Random** | the baseline all three must beat |

### 3 · Measure whether the choice helped

Train an identical small model on each subset. Show it 10 frames of where both hands have been
and ask it to predict their positions over the next 30 frames — one second. Score it by squared
error against what the person actually did, on clips the model never trained on.

The held-out clips come from **people and rooms absent from training**, so the score reflects
generalisation rather than memorisation.

---

## Results

Each method was compared against random selection using the same budget, the same held-out
clips, and the same model settings. 10 repeats × 2 budget sizes × 2 held-out conditions =
**40 paired comparisons** per method.

| Selection method | Comparisons won | Prediction error vs random |
|---|---|---|
| **Log-determinant** | **40 / 40** | **3.5% lower** |
| k-center | 40 / 40 | 3.4% lower |
| Facility location | 33 / 40 | 2.1% lower |
| Random, without the filtering step | 24 / 40 | no measurable difference |
| **Deliberately narrow** *(designed to fail)* | **1 / 40** | **5.8% higher** |

![how each method scored against random](reports/figs/paired.png)

### Why a 3.5% difference is credible

The margin is small, so the natural concern is that the measurement is too noisy to distinguish
anything and the result is chance. To test that, we included a selection **designed to fail**:
the same number of clips, drawn almost entirely from a handful of people in a handful of rooms.

It lost 39 of 40 comparisons. The EgoVerse authors had independently shown that variety in
demonstrators and scenes improves generalisation, and our measurement reproduces that finding.
It can therefore distinguish a good selection from a poor one, which makes the smaller margins
above measurements rather than noise.

### The result is not an artefact of our settings

We repeated the full comparison across nine configurations of the scoring model, varying its
capacity, regularisation, how much history it sees, and how far ahead it predicts. Nothing was
re-tuned to favour any method. The ordering held in every configuration: log-determinant beat
random in **9 of 9**, and the deliberately narrow selection in **0 of 9**.

---

## What this does not show

**Training on the full collection remains the best option.** It outperforms every subset we
selected, including the strongest one. Choosing well does not substitute for having more.

The useful claim is narrower: at a quarter of the data, a varied selection recovers roughly
**45%** of the difference between a random quarter and the full collection. That figure is what
matters when the full collection is not an option — deciding what to record next, what to pay
for annotation, or what will fit on a robot.

---

## Three findings about EgoVerse

**A "clip" is not a consistent unit across labs.** Median clip length is 90 seconds in one
collection and 11 seconds in another. In the largest collection, **1 in 7** clips is a complete
folding session stored as a single file. A request for "1,000 clips" therefore returns very
different quantities of footage depending on the source; requesting hours is safer.

We measured what those long clips contribute. Matched on **clip count**, removing them appears
harmful — 1.7% worse. That comparison is confounded: a single long clip contains roughly sixty
times the footage of a normal one, so the arm that retained them trained on twice as much
material. Matched on **footage**, removing them wins 10 comparisons out of 10, **4.7% better**.
The long clips were never better training data; there was simply more of it.

**Stored arrays are padded with blank frames.** Each clip is extended to a fixed block size, and
the true length is recorded separately. Read without trimming, every clip appears to have frozen
tracking at the end — which would have caused our checks to discard the entire library.

**Video is expensive; recorded positions are not.** The video for our test collection is 84.6 GB;
the recorded hand and head positions are 2.3 GB. Every signal here is computed from positions
alone, deliberately: a filtering step that costs more to run than the training it saves is not
worth running.

---

## Running it

EgoVerse publishes read-only credentials for public dataset access in its own README. Place them
in `~/.egoverse_aws_credentials`:

```
[default]
aws_access_key_id = <key from the EgoVerse README>
aws_secret_access_key = <secret from the EgoVerse README>
region = us-east-2
```

Then:

```bash
bash scripts/run_all.sh
```

This reproduces every number in this README from scratch in roughly 35 minutes, most of which is
downloading. Outputs are written to `reports/`.

---

## Repository layout

```
egoscore/
  access.py     connects to the EgoVerse database and object storage
  features.py   per-clip quality signals and the 41 descriptive features
  gate.py       the eight checks, their thresholds, and the rejection reasons
  select.py     the four selection methods, plus the one designed to fail
  proxy.py      the small model used to score each subset
scripts/        numbered and run in order; run_all.sh executes the full pipeline
reports/        results, figures, the full write-up, and the summary slide
demo/           the interactive demo and the slides
```

| Output | Contents |
|---|---|
| [`reports/keep_drop.csv`](reports/keep_drop.csv) | every clip, kept or rejected, with the reason |
| [`reports/validation.md`](reports/validation.md) | full write-up with all statistics |
| [`reports/egoscore_summary_slide.png`](reports/egoscore_summary_slide.png) | one-page summary |
| [`reports/results.csv`](reports/results.csv) | every comparison, unaggregated |

---

## Limitations

**The score is a proxy for policy quality, not robot success.** It measures how accurately a
model predicts human hand motion. It does not establish that a robot trained on the selected
subset completes more tasks. The EgoVerse authors use the same proxy for the same reason and say
so explicitly in their paper.

**The scoring model does not see the video.** This follows from the 84.6 GB figure above. A
vision-conditioned model might rank these subsets differently, and our measurement could not
detect that.

**One task, one collection.** We tested clothes-folding within a single lab's data. Transfer to
other tasks or collections is untested. That collection was not chosen because it favoured the
result — it is the only one that records which person filmed each clip and in which room, which
the held-out comparison requires.

**One check is measurable, one is not.** The filtering step demonstrably pays off where clips are
inconsistent (4.7% better on the largest collection). But the off-axis rule concerns how well the
task is *visible*, and since the scoring model does not use images, we cannot quantify its value.

**One check was wrong, and we caught it late.** That rule originally flagged clips where the hands
leave the frame. Three separate methods for determining that disagreed with what the frames
plainly showed — the camera is a fisheye at a steep angle, and standard projection models do not
handle it. The rule now claims only what is verifiable: the hands were held far off the camera's
centre line. `scripts/22_projection_check.py` and `scripts/24_hand_visibility.py` are the failed
attempts, retained rather than deleted.

**Our expected winner lost.** Facility location was the method we set out to use; log-determinant
and k-center both beat it. It remains in the results table.

---

## License

MIT
