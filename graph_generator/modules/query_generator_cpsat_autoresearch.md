# AutoResearch Program: Query Generator CPSAT

This document defines the research program for improving [`query_generator_cpsat.py`](/home/wangxingjian/DORO-STVG/graph_generator/modules/query_generator_cpsat.py).

It is intentionally narrow. It describes:

- what the module is supposed to do
- what success means
- what can be changed
- what must not be changed
- how to iterate

The style here is deliberate: the program should be easy to execute repeatedly, not impressive to read.

---

## Mission

Make the difficulty score emitted by this module performance-aligned.

In the final generated query set:

- lower `D` should usually correspond to better grounding performance
- higher `D` should usually correspond to worse grounding performance

Concretely, we want:

- lower `D` → higher `m_tIoU`, `m_vIoU`, `vIoU@0.3`, `vIoU@0.5`
- higher `D` → lower metrics
- 5-way difficulty buckets that are approximately monotone in model performance

This is a difficulty modeling task. It is not mainly:

- a query count optimization task
- a prompt polishing task
- an evaluation rewrite task

---

## Scope

Only edit:

- [`graph_generator/modules/query_generator_cpsat.py`](/home/wangxingjian/DORO-STVG/graph_generator/modules/query_generator_cpsat.py)

Do not solve this task by changing:

- evaluation logic
- model prompts in other modules
- downstream postprocessing in other files
- analysis scripts outside this module

Treat the graph input as fixed data:

- use `graph_generator/scene_graphs.jsonl`
- do not modify `graph_generator/scene_graphs.jsonl`

This module currently covers:

```text
graph
  → candidate interval construction
  → candidate difficulty estimation
  → CP-SAT clue/template selection
  → query.jsonl emission
```

This program is centered on candidate difficulty estimation and its interaction with sampling behavior.

---

## Primary Objective

The primary objective is not “make `D` look reasonable”. It is:

```text
Make D rank samples in the same direction as model difficulty.
```

This is the north star. All local improvements are subordinate to it.

The ideal end state is:

```text
very_easy > easy > medium > hard > very_hard
```

where “>`” means better model performance on average.

If the score is elegant but not performance-aligned, it is not good enough.
If the score is performance-aligned but implemented as a pile of hacks, it is also not good enough.

We need both:

- alignment with empirical performance
- simple, legible modeling

---

## Required Formula

Difficulty should not be defined from wording style or surface verbosity. A grounding query is difficult when the target is hard to distinguish from competing explanations in space and time.

From first principles, the relevant causes of difficulty are:

1. `Separability`
   How easily the target can be confused with competing objects or competing target assignments.

2. `Evidence Burden`
   How much evidence is required to uniquely identify the target.

3. `Temporal-Spatial Instability`
   How much the target state or its discriminative relations vary over time.

However, this program imposes a stricter modeling shape than a generic factor model.

The final difficulty must remain:

```text
D = λ * D_t + (1 - λ) * D_s
```

with exactly one hyperparameter:

- `λ`

Hard constraints:

- `D_t` is one single temporal difficulty term
- `D_s` is one single spatial difficulty term
- neither `D_t` nor `D_s` may internally be defined as a sum of multiple sub-terms
- additive composition is allowed only once, at the top level in `D`
- `D_t`, `D_s`, and `D` must each lie in `[0, 1]`

This means:

- you may redesign what `D_t` means
- you may redesign what `D_s` means
- you may use bounded nonlinear transforms inside each branch
- you may not define `D_t = a + b`
- you may not define `D_s = c + d`
- you may not introduce more free weights inside the branches

Allowed shape examples:

```text
D_t = g(temporal-instability)
D_s = h(spatial-confusability)
```

or

```text
D_t = g(required temporal evidence)
D_s = h(target-distractor separability)
```

Not allowed:

```text
D_t = a * change + b * time_span
D_s = c * overlap + d * distractors
```

The two branches should each correspond to one irreducible source of difficulty, and the only tunable tradeoff should be the temporal-spatial balance at the top level.

---

## Current Situation

The current implementation computes:

- `D_t` from action/relation change counts and temporal span overlap counts
- `D_s` from same-class distractor counts and interval tIoU
- `D = lambda * D_t + (1 - lambda) * D_s`

This has good properties:

- compact
- deterministic
- cheap to compute

But it is currently insufficient because:

- it measures candidate structure more than actual model confusion
- it does not guarantee that each branch is one clean irreducible factor
- it does not explicitly model target separability under shared clue sets
- it entangles difficulty estimation with template and sampling effects
- fixed bucket thresholds are not calibrated against performance

---

## What Good Looks Like

A good solution has all of the following properties:

1. Continuous difficulty behaves sensibly
   `corr(D, metric)` is stably negative for key metrics.

2. Buckets behave sensibly
   5-way buckets are approximately monotone in model performance.

3. Single-target and multi-target behavior are both reasonable
   The score should not work only for one arity regime.

4. The implementation is short and conceptually clean
   We should be able to explain the difficulty model on one whiteboard.

5. The module exports interpretable metadata
   Intermediate components are visible in `cand.meta` or equivalent local structure.

---

## What Not To Do

Do not:

- add case-by-case heuristics for specific templates
- patch bucket thresholds repeatedly without fixing the underlying score
- encode template names as latent difficulty truth
- solve monotonicity by changing downstream evaluation
- add many weak features just because they are available
- make the code longer every round while only slightly moving the metric

This program values parsimony. A smaller, more principled model is preferred over a larger, more brittle one.

---

## Concrete Procedure

Use this exact operational path.

### Fixed input

Use:

- `graph_generator/scene_graphs.jsonl`

Do not modify it during this research.

### Query generation environment

Use:

- `envs/graph_generator/main/.venv`

From repo root:

```bash
cd DORO-STVG
source envs/graph_generator/main/.venv/bin/activate
cd graph_generator
```

Generate queries by following the path used in [`run_generator.sh`](/home/wangxingjian/DORO-STVG/graph_generator/scripts/run_generator.sh):

```bash
python -m modules.query_generator_cpsat \
  --input_path scene_graphs.jsonl \
  --output_path output/query.jsonl \
  --time_limit_sec 3.0 \
  --seed 7 \
  --use_llm_polish True \
  --polish_model_name gemini-3-flash-preview \
  --max_concurrent_per_key 100 \
  --max_retries 5
```

This writes:

- `graph_generator/output/query.jsonl`

### Formatting

After generation, run:

```bash
bash scripts/run_formatted.sh
```

This writes:

- `graph_generator/output/query_train.jsonl`

### Evaluation environment

Use:

- `envs/eval/.venv`

From repo root:

```bash
deactivate || true
source envs/eval/.venv/bin/activate
cd eval
```

Run:

```bash
bash run_eval.sh
```

### Important path note

There is a current path mismatch:

- generation writes `graph_generator/output/query.jsonl`
- formatting writes `graph_generator/output/query_train.jsonl`
- [`eval/run_eval.sh`](/home/wangxingjian/DORO-STVG/eval/run_eval.sh) currently reads `graph_generator/output/query.jsonl`

So if evaluation must use the formatted file, then before running eval you must do one of:

1. change `ANNOTATION_PATH` in `eval/run_eval.sh` to `graph_generator/output/query_train.jsonl`
2. or copy the formatted file over `graph_generator/output/query.jsonl`

The first option is cleaner because it preserves both raw and formatted outputs.

### Minimal end-to-end sequence

```bash
cd DORO-STVG
source envs/graph_generator/main/.venv/bin/activate
cd graph_generator
python -m modules.query_generator_cpsat \
  --input_path scene_graphs.jsonl \
  --output_path output/query.jsonl \
  --time_limit_sec 3.0 \
  --seed 7 \
  --use_llm_polish True \
  --polish_model_name gemini-3-flash-preview \
  --max_concurrent_per_key 100 \
  --max_retries 5
bash scripts/run_formatted.sh
deactivate || true
cd ..
source envs/eval/.venv/bin/activate
cd eval
bash run_eval.sh
```

---

## Operating Loop

```text
START:
1. Look under eval/res/ and find the latest evaluation result directory.
2. Read its summary and detailed outputs.
3. Join the latest evaluation outputs with the generated query difficulty fields.
4. Analyze difficulty against metrics:
   - corr(D, m_tIoU)
   - corr(D, m_vIoU)
   - corr(D, vIoU@0.3)
   - 5-bucket monotonicity
   - per-bucket sample counts
   - single-target vs multi-target behavior
   - different difficulty levels against metrics
5. Decide whether this is a baseline run or a continuation run:
   - if no optimization history exists, mark the latest analyzed run as baseline
   - otherwise continue from the latest analyzed best run

LOOP:
1. Generate query.jsonl with the current module.
2. Format the generated file.
3. Evaluate with the fixed evaluation pipeline.
4. Join query outputs with evaluation outputs.
5. Measure:
   - corr(D, m_tIoU)
   - corr(D, m_vIoU)
   - corr(D, vIoU@0.3)
   - 5-bucket monotonicity
   - per-bucket sample counts
   - single-target vs multi-target behavior
   - different difficulty levels against metrics
6. Inspect failures:
   - low D but low score
   - high D but high score
7. Form one hypothesis.
8. Make one coherent change in query_generator_cpsat.py.
9. Check stopping conditions.
10. Repeat unless one stopping condition is met.

END:
1. Output a summary report.
2. The report must include:
   - baseline → best metric changes
   - what each round changed and how effective it was
   - which version is the final best_version
```

The START step is mandatory. Do not begin editing blindly.

The rule is one main idea per iteration. If multiple things are changed at once, attribution becomes weak and the research loop degrades.

For quick research judgment:

- around 20 to 30 samples per difficulty level is usually enough to detect whether the trend is roughly right
- this is enough for directional judgment, not final statistical certification
- if a bucket has fewer than ~20 samples, treat its mean as weak evidence
- if all 5 buckets have at least ~20 to 30 samples and the ordering is still not visible, the difficulty design is probably wrong

So the immediate target is not perfectly balanced large buckets. It is enough support in each level to reveal whether the ranking law is working.

### Stopping conditions

- success stop:
  there is a clear and stable correlation between difficulty and model performance, and the target is achieved
- convergence stop:
  5 consecutive rounds show no improvement over the current best result
- budget stop:
  20 optimization rounds have been used, excluding the baseline run

---

## Diagnostic Questions

When a low-`D` sample still performs badly, ask:

- Was it actually poorly separable?
- Did the score ignore the need for relational evidence?
- Did the score treat a multi-target sample like independent single targets?
- Did the selected clues create hidden ambiguity that the difficulty score never modeled?

When a high-`D` sample still performs well, ask:

- Is the score over-penalizing benign motion or harmless distractors?
- Is the target actually obvious despite structural richness?
- Did template structure inflate apparent difficulty without increasing grounding difficulty?

These questions should drive the next change.

---

## Design Rules

### Allowed kinds of changes

In scope:

- redesigning candidate difficulty decomposition
- redefining `D_t` and `D_s`
- changing how difficulty buckets are assigned
- changing candidate ordering logic when it leaks non-difficulty effects into the emitted difficulty distribution
- exposing better introspection statistics in local metadata

Conditionally allowed:

- changing template weights or template bucket usage, but only to decouple template coverage from difficulty semantics

Out of scope:

- changing downstream eval metric definitions
- changing LLM polishing prompts to artificially sharpen bucket separability
- adding post hoc relabeling outside this file

### What not to do

Do not:

- add case-by-case heuristics for specific templates
- patch bucket thresholds repeatedly without fixing the underlying score
- encode template names as latent difficulty truth
- solve monotonicity by changing downstream evaluation
- add many weak features just because they are available
- make the code longer every round while only slightly moving the metric

This program values parsimony. A smaller, more principled model is preferred over a larger, more brittle one.

### Metadata requirement

Every meaningful difficulty component should be inspectable. The module should make it easy to analyze why a candidate got its score.


### Simplicity criterion

A proposed change should be rejected if:

- it increases code size significantly
- it adds many thresholds or special cases
- it cannot be summarized clearly
- it improves one bucket but damages the overall semantics of difficulty

A proposed change should be favored if:

- it removes ad hoc logic
- it makes the score easier to interpret
- it improves continuous ranking quality
- it improves bucket monotonicity without brittle hacks

Top-tier design here means the model of difficulty feels inevitable in hindsight.

---

## Clean Code Requirement

This module already shows signs of local overgrowth. Future iterations should push it toward cleaner structure, not just better metrics.

Use Python clean-code guardrails as the implementation standard:

- functions should stay short and single-purpose
- avoid deep nesting; prefer guard clauses
- avoid scattering difficulty semantics across unrelated helper paths
- do not introduce broad exception handling or silent fallbacks
- do not keep dead intermediate abstractions after a redesign
- tests are required for any behavior change in the difficulty logic

If a change improves the score slightly but makes [`query_generator_cpsat.py`](/home/wangxingjian/DORO-STVG/graph_generator/modules/query_generator_cpsat.py) longer, harder to read, and more heuristic-driven, it should usually be rejected.

Preferred direction:

- fewer concepts
- fewer branches
- fewer magic constants
- clearer names
- shorter difficulty code

---

## Success Criteria

This program is successful only if most of the following become true at the same time:

1. `D` has stable negative association with key grounding metrics.
2. 5-way buckets are substantially more monotone than they are now.
3. Bucket behavior is not explained away by template distribution alone.
4. Single-target and multi-target settings both remain coherent.
5. Single-target grounding is generally easier than multi-target grounding.
6. The difficulty code in [`query_generator_cpsat.py`](/home/wangxingjian/DORO-STVG/graph_generator/modules/query_generator_cpsat.py) becomes cleaner or at least not worse.
7. The final formula still has the form `D = λ·D_t + (1-λ)·D_s`, with no additive decomposition inside `D_t` or `D_s`.

If the metrics improve but the code becomes a patchwork, keep going.
If the code becomes elegant but the score still does not track performance, keep going.
Only stop when both the science and the implementation are good.
