# Held-out run on fresh prompts (frozen configuration)

`FROZEN.md` records the configuration that was fixed before any fresh prompt was run. Thresholds are fitted on the original 787
prompts only (raw score cut-offs); the 614 fresh prompts are used for measurement, once.

Commands (run from the repository root; steps 1-3 need an API key and cost roughly USD 5-8; every collection step is resumable):

    python src/build_fresh.py --old data/data_final.jsonl --out data/data_fresh.jsonl --mmlu_pro 300 --gsm8k 100 --seed 2026
    python src/collect.py --data data/data_fresh.jsonl --out cache/responses_fresh.jsonl --workers 8
    python src/collect_samples.py --data data/data_fresh.jsonl --out cache/samples_fresh.jsonl --model openai/gpt-oss-120b --k 2 --temperature 0.8 --max_tokens 12000
    python src/selfcons_scores.py --data data/data_fresh.jsonl --responses cache/responses_fresh.jsonl --samples cache/samples_fresh.jsonl --out scores/selfcons_fresh.jsonl
    python src/make_signal_scores.py --data data/data_fresh.jsonl --responses cache/responses_fresh.jsonl --signal disagree --models openai/gpt-oss-120b,openai/gpt-oss-20b --out scores/dis_fresh.jsonl
    python src/judge_scores.py --data data/data_fresh.jsonl --out scores/judge_fresh.jsonl --judge meta-llama/llama-3.3-70b-instruct
    python src/collect_extra.py --data data/data_final.jsonl --out_prefix scores/extra_old
    python src/collect_extra.py --data data/data_fresh.jsonl --out_prefix scores/extra_fresh
    python src/embed_scores.py --old data/data_final.jsonl --old_responses cache/responses_final.jsonl --fresh data/data_fresh.jsonl --out_old scores/embed_old.jsonl --out_fresh scores/embed_fresh.jsonl

Evaluation (no API key needed; reproduces `results/fresh_summary.csv`):

    python src/eval_fresh.py --old_data data/data_final.jsonl --old_resp cache/responses_final.jsonl \
      --fresh_data data/data_fresh.jsonl --fresh_resp cache/responses_fresh.jsonl \
      --scorers selfcons3=scores/selfcons_k2.jsonl:scores/selfcons_fresh.jsonl,disagree=scores/disagree_120b.jsonl:scores/dis_fresh.jsonl,judge=scores/judge.jsonl:scores/judge_fresh.jsonl,logprob=scores/extra_old_logprob.jsonl:scores/extra_fresh_logprob.jsonl,verbal=scores/extra_old_verbal.jsonl:scores/extra_fresh_verbal.jsonl,embed=scores/embed_old.jsonl:scores/embed_fresh.jsonl \
      --out runs/fresh_results

The log-probability probe is skipped automatically: 36% of fresh prompts (41% of the original ones) returned no logprobs from the provider.
