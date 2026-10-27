# Routing prompts between a base and a strong LLM

Code, cached model answers and analysis scripts for the paper

> **Routing Prompts Between a Base and a Strong LLM: How Much Do Lightweight Difficulty Signals Save?**
> Dhilip (2026). `paper/main.tex`, compiled copy in `paper/main.pdf`.

We route each prompt between a base model (`openai/gpt-oss-120b`) and a strong model (`anthropic/claude-sonnet-5.5`). Seven models answered every
prompt once; routers are then simulated **offline** on the cached answers and costs, so everything below can be reproduced **without an API key**.
Each scorer is charged for its own calls.

## Quick start (no API key needed)

```bash
git clone <this repository> && cd <repo>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. 8-split table of the paper (95% target, full set)
cd analysis && python repro.py && cd ..

# 2. held-out run on the 614 fresh prompts (reproduces results/fresh_summary.csv)
python src/eval_fresh.py --old_data data/data_final.jsonl --old_resp cache/responses_final.jsonl \
  --fresh_data data/data_fresh.jsonl --fresh_resp cache/responses_fresh.jsonl \
  --scorers selfcons3=scores/selfcons_k2.jsonl:scores/selfcons_fresh.jsonl,disagree=scores/disagree_120b.jsonl:scores/dis_fresh.jsonl,judge=scores/judge.jsonl:scores/judge_fresh.jsonl,logprob=scores/extra_old_logprob.jsonl:scores/extra_fresh_logprob.jsonl,verbal=scores/extra_old_verbal.jsonl:scores/extra_fresh_verbal.jsonl,embed=scores/embed_old.jsonl:scores/embed_fresh.jsonl \
  --out runs/fresh_results

# 3. 100-split tables, bootstrap intervals, threshold rules (writes rules_100splits.csv, boot_lift.csv; gen_tables.py rewrites paper/tex/)
cd analysis && python main_an.py && python boot.py && python gen_tables.py
```

Expected for (1): self-consistency 50.65 % savings, -3.36 pp accuracy, 8/8 splits at target, lift +3.28; disagreement 48.69 / -4.28 / +2.15;
learned combiner 48.95 / -4.09 / +2.36.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Pipeline: `make_dataset.py` (prompt set), `collect.py` (answers from OpenRouter), `collect_samples.py` (extra samples), score builders (`judge_scores.py`, `make_signal_scores.py`, `selfcons_scores.py`, `combo_scores.py`, `embed_scores.py`, `collect_extra.py`), `simulate.py` (offline router simulation), `run_seeds.py`, `pair_study.py`, `eval_fresh.py` (frozen-configuration evaluation) |
| `data/` | Prompt files: `data_final.jsonl` (787 analysed prompts), `data_final_cat.jsonl` (+ MMLU-Pro subject), `data_fresh.jsonl` (fresh held-out prompts) |
| `cache/` | Cached answers of the seven models (`responses_*.jsonl`) and the extra gpt-oss-120b samples (`samples_*.jsonl`) |
| `scores/` | Per-prompt difficulty scores and their overhead cost (judge, disagreement, self-consistency, combiners, embedding router, verbalised confidence, log-probabilities) |
| `analysis/` | Scripts behind the 100-split tables, bootstrap intervals, threshold-rule comparison and signal table (`core.py` is shared code) |
| `results/` | Per-seed simulation summaries (8 splits), pair study, fresh-run summary and the analysis outputs used in the paper |
| `fresh/` | `FROZEN.md` (the configuration fixed before the fresh run) and the commands of the fresh run |
| `paper/` | LaTeX source, tables, figures, `build_assets.py` (rebuilds `results/main_results_8seeds.csv` and the figures from `results/seeds_*`) |

Score files use one JSON object per line: `{"id", "score", "cost", "reuse"?}`. A higher score means "harder"; `cost` is the router overhead that
`simulate.py` charges, and `reuse` marks a base-model answer that the probe already paid for (`--credit_reuse`).
Which file is which in the paper: `selfcons_k2` = self-consistency (3 votes); `disagree_120b` = disagreement gpt-oss-120b vs gpt-oss-20b;
`combo_120b` = "Learned combiner"; `combo_sc` / `combo_sc_cat` = combiner with samples (and subject); `judge` = LLM judge (llama-3.3-70b);
`disagree_llama8b_oss20b` = an early variant that is not used in the paper's tables.

## Re-collecting the answers (needs an API key, costs money)

Create an [OpenRouter](https://openrouter.ai) key and **export it as an environment variable**; do not write it into any tracked file:

```bash
export OPENROUTER_API_KEY=sk-or-...        # Windows PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
python src/make_dataset.py --help           # rebuild the prompt set (needs `datasets` and internet access)
python src/collect.py --data data/data_final.jsonl --out cache/responses_final.jsonl --workers 8
```

The scripts also accept an untracked `openrouter_key.txt` next to them; it is listed in `.gitignore`. Model slugs, providers and prices change; the
cached answers were collected on specific dates and prices are an OpenRouter snapshot. The full set of commands for each stage is in the docstring
at the top of each script, and `fresh/README.md` has the commands of the held-out run (paths there match this layout).

## Notes

* All scripts run from the **repository root** unless stated otherwise (`analysis/` scripts run from inside `analysis/`).
* `results/seeds_*` hold only each seed's `summary.csv`; the bucket plots and frontier tables of the original runs are regenerated by `simulate.py`.
* The cached answers are model outputs for public benchmark prompts (GSM8K, MATH-500, MMLU-Pro, AIME 2024/2025). The benchmarks keep their own licences and the model
  outputs are subject to the providers' terms; check them before reusing the data for anything beyond reproducing this study.
* Code is released under the MIT licence (`LICENSE`).

## Citation

```bibtex
@misc{dhilip2026routing,
  author = {Dhilip},
  title  = {Routing Prompts Between a Base and a Strong {LLM}: How Much Do Lightweight Difficulty Signals Save?},
  year   = {2026},
  note   = {Code and data: <repository URL>}
}
```
