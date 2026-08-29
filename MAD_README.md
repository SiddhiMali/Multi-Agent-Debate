# Multi-Agent Chain on GSM8K — What This Notebook Does

`MAD.ipynb` runs a **sequential 4-agent chain** that solves grade-school math word
problems (the GSM8K benchmark) and studies **where a chain breaks** when each agent
trusts the one before it. It is not a debate — the agents never argue or vote. They
form an assembly line.

## The architecture

```
   Extractor  ─▶  Formulator  ─▶  Solver  ─▶  Verifier  ─▶  final answer
  (givens+goal)   (equations)    (number)   (sanity-check)
```

Each agent has one fixed job and consumes the previous agent's output **as if it were
correct**:

| Agent | Job | Explicitly told |
|---|---|---|
| **Extractor** | List the known quantities and what's being asked. Does *not* solve. | — |
| **Formulator** | Turn the extraction into equations. Does *not* compute. | "assume the extraction is correct" |
| **Solver** | Follow the equations to a number. | "assume the equations are correct" |
| **Verifier** | Sanity-check and state the final answer. | given the proposed solution |

## The core idea being tested

The design deliberately gives the chain **no group error-correction**. Each agent
trusts its input blindly, so an early mistake propagates downstream and compounds.
Because roles are fixed, when the final answer is wrong you can open the transcript and
see *which stage* introduced the error. The author's own comment frames this fragility
as "the finding, not a bug" — the experiment is about locating and understanding failure
in a dependency chain, not just maximizing accuracy.

## Why GSM8K

GSM8K answers are single numbers with published ground truth (formatted as `#### 42`),
so correctness is checked by exact match — no LLM-judge, no subjective scoring, no
calibration needed. You get a hard accuracy number and can trust it.

## The setup

- **Models (4-bit quantized, loaded via HuggingFace `transformers`):**
  - Extractor, Formulator, Verifier → **Qwen2.5-7B-Instruct**
  - Solver → **Mistral-7B-Instruct-v0.2**
  - This "capability-diverse" split is a study knob: putting a different model at one
    role lets you test whether a weaker agent at a given stage poisons the chain.
- **Runs on Colab with a GPU.** Two 7B models in 4-bit fit a free T4.
- **60 problems**, sampling do_sample=True at temperature 0.7 (so runs vary slightly).
- Saves full transcripts to Google Drive as `chain_gsm8k_transcripts.json`.

## Results from the run in this notebook

**Accuracy: 48 / 60 correct = 80%.**

The 12 failures (gold vs the chain's answer):

| Problem | Gold | Chain | Note |
|---|---|---|---|
| P02 | 70000 | 95000 | Verifier *overrode a solution and introduced a new error* |
| P06 | 260 | 240 | error upstream of the Solver |
| P07 | 160 | 420 | Verifier changed the answer at the last step |
| P08 | 45 | 135 | multi-step motion problem |
| P12 | 13 | 12 | off-by-small |
| P13 | 18 | None | no parseable number produced |
| P20 | 15 | 11 | — |
| P21 | 14 | 2 | large deviation |
| P24 | 26 | 26.67 | rounding / integer-vs-float |
| P25 | 2 | None | no parseable number produced |
| P31 | 80 | 81 | off-by-one |
| P41 | 200 | 1200 | order-of-magnitude error |

### What the failure transcripts reveal (the actual payoff)

The whole point of the fixed-role design is being able to attribute each failure to a
stage. Two patterns stand out in the saved transcripts:

- **The Verifier sometimes *breaks* a correct chain.** In P02 the Solver produced a
  clean computed result, but the Verifier "sanity-checked" it, talked itself into a
  different figure, and emitted a *worse* answer (95000). In P07 the Verifier again
  overrode the Solver at the final step. The last agent, meant as a safety net, is
  itself a source of error — a concrete, non-obvious finding.
- **Two failures (P13, P25) produced no parseable number at all** — the answer
  extraction returned `None`. That's a *format/parsing* failure, not a reasoning
  failure, and it's a distinct category worth separating from wrong-number cases.

This is exactly what the experiment is for: not "the chain got 80%", but "*of the 20%
it missed, here is where and how it broke.*"

## The two cells

1. **Main pipeline** — config, model loading, the four role prompts, `run_chain()`
   (Extractor → Formulator → Solver → Verifier), the GSM8K loop over 60 problems, and
   the accuracy report. Optionally closes the loop (`close_loop=True`) to feed the
   Verifier's answer back for a consistency check.
2. **Failure analysis** — reopens the saved transcripts and prints every *wrong* case
   with all four stages, so you can eyeball which stage introduced the error.

## How to run it (Colab)

1. Runtime → Change runtime type → **GPU**.
2. Install deps:
   ```python
   !pip install transformers accelerate bitsandbytes datasets torch
   ```
3. Run the main cell. It downloads both models and GSM8K automatically.
4. Run the analysis cell **after** the first cell has saved transcripts.

**Config knobs worth knowing** (top of the main cell):
- `n_problems` — drop to `5` for a quick smoke-test before a full 60-problem run.
- `save_to_drive` — set `False` to skip the Google Drive mount/auth prompt.
- `role_models` — reassign which model plays which role to study weak-agent effects.
- `close_loop` — `True` adds the Verifier→Extractor consistency feedback step.

Note: because sampling is stochastic (temperature 0.7), re-running won't reproduce the
exact 80% — expect small variation. For reproducible numbers, set `do_sample=False` in
`generate()` (greedy decoding).

## Ideas to extend

- **Fix the Verifier.** Since it *introduces* errors here, try prompting it to only
  flag/return the Solver's answer unless it can show a specific arithmetic error — and
  measure whether accuracy rises.
- **Separate error types in the report.** Split "wrong number" from "no parseable
  answer" (the `None` cases) so parsing bugs don't masquerade as reasoning failures.
- **Per-stage attribution at scale.** Auto-label each wrong case by first-broken stage
  instead of reading transcripts by hand.
- **Swap the weak-role model** and re-run to see if the break point moves.
