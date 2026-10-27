Frozen before any fresh prompt was run (see the paper, Section 3.8).

Models: base = openai/gpt-oss-120b, strong = anthropic/claude-sonnet-5.5
Self-consistency: greedy answer + 2 samples, temperature 0.8 (3 votes)
Disagreement: gpt-oss-120b vs openai/gpt-oss-20b
LLM judge model: meta-llama/llama-3.3-70b-instruct (same as the original judge.jsonl)
Quality target: 95% of Sonnet's accuracy
Safety margin: report both 0 and 0.02
Thresholds: fitted on the OLD 787 prompts only; never on the fresh prompts
Reporting: run the evaluation once and report whatever comes out
