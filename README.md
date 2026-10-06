# Auditable Explanations for Financial-Crime Compliance

Code for a matched-twin benchmark of **auditable explanations** across anti-money-laundering (AML) and market abuse. Each scenario is scored against a four-element rubric — **mechanism**, the **rule that is defeated** (loophole in AML, breach in market abuse), the **distinguishing** facts that separate a violation from its innocent twin, and the **decision** (REPORT / PROCEED / CLEAR) — rather than a yes/no label.

*Anonymous repository for peer review. It contains no real customer, transaction, or account data; every scenario is synthetic.*

The dataset (10,057 matched pairs = 20,114 scenarios, plus expert seeds) is released separately: **https://huggingface.co/datasets/Anonymousbabygoat/auditable-fincrime-benchmark**.

## Pipeline

1. **Expert seeds.** A qualified Singapore compliance lawyer authored one matched pair per typology — a *violation* and an *innocent twin* — with a four-element answer key for each. `seeds_aml.json` (120 typologies: 90 seen / 30 held-out) and `seeds_market.json` (100: 75 / 25). Each record has `id, family, typology, split, vstory, vrubric, tstory, trubric`.

2. **Scaling with automatic checks.** `scaler_aml.py` / `scaler_market.py` few-shot a teacher model (Claude Sonnet 5) from each **training** seed to generate new matched pairs on rotating diversity axes, and run automatic checks on every generated pair:
   - schema completeness (all four elements present);
   - a **give-away-word guard** on the story bodies (`LEAK_WORDS`), so the neutral record never names the abuse;
   - a **near-duplicate guard** — flag if string similarity to the seed `> 0.85`, or to an earlier variant of the same typology `> 0.75`.
   Held-out typologies are never generated. Flagged rows are regenerated (`SC_REDO_FLAGGED=1`). The API key is read from the environment and never stored.

   ```bash
   export SC_API_KEY=...            # teacher-model key (not stored)
   export SC_BASE=https://api.anthropic.com/v1/   SC_MODEL=claude-sonnet-5
   export SC_SEEDS=seeds_aml.json   SC_A_PER=56   SC_OUT=AML90_scaled_FULL.csv
   python3 scaler_aml.py            # market: scaler_market.py, SC_SEEDS=seeds_market.json, SC_A_PER=67
   ```

3. **Fine-tuning.** `train_aml_sft.py` / `train_market_sft.py` train a LoRA adapter (rank 16, α=32, dropout 0.05) on the attention and MLP projections for three epochs with frozen base weights (transformers + peft only). PBS launchers (`run_aml_sft.pbs`, `run_market_eval_qwen7b.pbs`, `download_models.pbs`); the SFT target is built from the rubric so the model learns to emit the four elements.

4. **Evaluation.** `eval_aml90_2way.py` / `eval_market_2way.py` run base vs. LoRA on the **held-out** typologies (two-way: violation + innocent twin), writing per-scenario answers to CSV. Seeds are loaded from the JSON files (`EVAL_SEEDS`).

5. **Grading.** Each reasoning element is graded in its own call by a judge model (Claude Sonnet 5), with the decision checked directly against the answer key; a scenario is *fully correct* only when all three reasoning elements pass and the decision is right. *(Per-element grader script to be added.)*

6. **Prompting frontier models + APO.** An innocent-first prompt (Contrast in AML, Innocent-Hypothesis-First in market abuse) builds the innocent twin before naming the distinguishing facts; automatic prompt optimization (DSPy MIPROv2) is run from a bare seed as a comparison. *(APO scripts to be added.)*

## Contents

| File | Purpose |
|---|---|
| `scaler_aml.py`, `scaler_market.py` | Teacher-model scaling + automatic checks |
| `seeds_aml.json`, `seeds_market.json` | Expert seed pairs (with train/held-out split) |
| `train_aml_sft.py`, `train_market_sft.py` | LoRA SFT |
| `eval_aml90_2way.py`, `eval_market_2way.py` | Two-way held-out evaluation |
| `run_*.pbs`, `download_models.pbs` | Cluster launchers (representative) |

## Requirements

Generation/eval harness: `openai` (used as the HTTP client for the teacher/judge). Fine-tuning: `torch`, `transformers`, `peft`, `accelerate`. Python 3.10+.

## Ethics

The benchmark is built only from synthetic, expert-authored and model-grown scenarios; suspicious-activity and suspicious-transaction reports and the underlying records are confidential by law and are never used. The models and data are intended to assist a qualified analyst, not to file reports autonomously.
