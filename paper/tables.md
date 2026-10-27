### Table A. Reference policies (full set, mean of 8 splits)
| Policy | Accuracy % | USD per 100 prompts | Savings vs Sonnet % |
|---|---|---|---|
| Always gpt-oss-120b | 79.3 | 0.016 | 95.1 |
| Always Claude Sonnet 5.5 | 91.8 | 0.333 | 0.0 |
| Always Claude Opus 5.5 | 94.5 | 0.738 | -121.8 |
| Oracle router (cheapest correct of the two) | 93.8 | 0.084 | 74.8 |

### Table B. Reference policies (MMLU-Pro only)
| Policy | Accuracy % | USD per 100 prompts | Savings vs Sonnet % |
|---|---|---|---|
| Always gpt-oss-120b | 76.2 | 0.011 | 96.5 |
| Always Claude Sonnet 5.5 | 89.6 | 0.297 | 0.0 |
| Always Claude Opus 5.5 | 93.0 | 0.638 | -114.8 |
| Oracle router (cheapest correct of the two) | 92.1 | 0.073 | 75.5 |

### Table C. Routers at the 95% operating point, full set
| Router | Savings vs Sonnet % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift over random mix (pp) |
|---|---|---|---|---|---|
| Self-consistency (3 votes) | 50.6 (3.2) | -3.4 | 96.3 | 8/8 | 3.3 (0.8) |
| Disagreement (120b vs 20b) | 48.7 (3.0) | -4.3 | 95.3 | 5/8 | 2.1 (0.9) |
| Learned combiner | 49.0 (2.3) | -4.1 | 95.5 | 4/8 | 2.4 (0.6) |
| Combiner + samples | 45.4 (8.8) | -5.0 | 94.5 | 6/8 | 1.0 (1.1) |
| Combiner + samples + subject | 44.3 (5.9) | -4.4 | 95.2 | 5/8 | 1.4 (0.8) |
| Prompt length | 54.6 (13.9) | -5.1 | 94.5 | 3/8 | 2.3 (0.5) |
| TF-IDF ridge | 46.7 (11.8) | -5.6 | 94.0 | 3/8 | 0.7 (0.6) |
| Heuristic features | 29.5 (3.6) | -4.8 | 94.8 | 3/8 | -0.9 (1.1) |
| Oracle difficulty (ceiling) | 58.2 (3.2) | -4.0 | 95.6 | 5/8 | 3.7 (0.7) |
| Random scores | 43.8 (10.8) | -5.9 | 93.6 | 1/8 | -0.1 (0.9) |

### Table D. Routers at the 95% operating point, MMLU-Pro only
| Router | Savings vs Sonnet % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift over random mix (pp) |
|---|---|---|---|---|---|
| Self-consistency (3 votes) | 56.6 (3.9) | -4.5 | 94.9 | 4/8 | 3.3 (0.8) |
| Disagreement (120b vs 20b) | 54.4 (11.8) | -6.0 | 93.3 | 1/8 | 1.6 (1.4) |
| Learned combiner | 49.6 (8.2) | -4.3 | 95.2 | 4/8 | 2.6 (1.0) |
| Combiner + samples | 52.2 (2.9) | -5.2 | 94.1 | 1/8 | 2.0 (0.7) |
| Combiner + samples + subject | 53.0 (3.0) | -5.5 | 93.9 | 1/8 | 1.9 (0.9) |
| Prompt length | 53.6 (14.8) | -5.5 | 93.9 | 2/8 | 2.1 (1.4) |
| TF-IDF ridge | 51.5 (11.3) | -6.0 | 93.3 | 2/8 | 1.1 (1.1) |
| Heuristic features | 25.3 (7.1) | -4.6 | 94.9 | 5/8 | -1.0 (1.4) |
| Oracle difficulty (ceiling) | 57.0 (7.4) | -4.4 | 95.1 | 5/8 | 3.5 (0.9) |
| Random scores | 26.7 (10.4) | -4.1 | 95.4 | 6/8 | -0.3 (1.1) |

### Table E. Routers at the 99% operating point, full set
| Router | Savings vs Sonnet % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift over random mix (pp) |
|---|---|---|---|---|---|
| Self-consistency (3 votes) | -14.3 (0.6) | +0.0 | 100.0 | 8/8 | -1.9 (0.2) |
| Disagreement (120b vs 20b) | 7.7 (7.0) | -1.3 | 98.6 | 2/8 | -0.4 (0.4) |
| Learned combiner | 13.4 (13.2) | -1.4 | 98.5 | 3/8 | 0.4 (1.1) |
| Combiner + samples | -0.6 (8.2) | -0.9 | 99.0 | 4/8 | -1.0 (0.5) |
| Combiner + samples + subject | -2.3 (3.8) | -0.7 | 99.2 | 7/8 | -1.0 (0.3) |
| Prompt length | 12.4 (6.8) | -0.8 | 99.1 | 5/8 | 0.9 (0.3) |
| TF-IDF ridge | 14.9 (9.7) | -1.2 | 98.7 | 3/8 | 0.8 (0.5) |
| Heuristic features | 7.0 (4.8) | -1.0 | 99.0 | 5/8 | -0.0 (0.6) |
| Oracle difficulty (ceiling) | 33.7 (8.8) | -1.2 | 98.7 | 6/8 | 3.3 (0.6) |
| Random scores | 5.7 (6.8) | -1.0 | 98.9 | 4/8 | -0.3 (0.6) |

### Table F. Routers at the 99% operating point, MMLU-Pro only
| Router | Savings vs Sonnet % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift over random mix (pp) |
|---|---|---|---|---|---|
| Self-consistency (3 votes) | -10.7 (0.5) | +0.0 | 100.0 | 8/8 | -1.5 (0.1) |
| Disagreement (120b vs 20b) | 10.2 (9.1) | -1.3 | 98.6 | 3/8 | 0.2 (0.6) |
| Learned combiner | 14.3 (13.5) | -1.4 | 98.4 | 3/8 | 0.6 (1.0) |
| Combiner + samples | 7.4 (13.9) | -1.6 | 98.2 | 2/8 | -0.5 (0.9) |
| Combiner + samples + subject | 3.6 (7.6) | -1.2 | 98.7 | 4/8 | -0.6 (0.6) |
| Prompt length | 11.1 (3.4) | -0.4 | 99.5 | 6/8 | 1.1 (0.4) |
| TF-IDF ridge | 15.5 (6.8) | -1.5 | 98.3 | 5/8 | 0.7 (0.8) |
| Heuristic features | 7.6 (6.1) | -1.1 | 98.8 | 4/8 | -0.0 (0.5) |
| Oracle difficulty (ceiling) | 31.7 (2.6) | -0.9 | 99.0 | 4/8 | 3.5 (0.4) |
| Random scores | 0.8 (2.3) | -0.2 | 99.8 | 7/8 | -0.0 (0.1) |

### Table G. Number of votes, full set
| Probe | Target | Savings % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift (pp) |
|---|---|---|---|---|---|---|
| 2 votes (K=1) | 95% | 35.3 (37.3) | -3.6 | 96.1 | 3/8 | 1.5 (2.3) |
| 2 votes (K=1) | 99% | -9.7 (0.4) | +0.0 | 100.0 | 8/8 | -1.3 (0.2) |
| 3 votes (K=2) | 95% | 50.6 (3.2) | -3.4 | 96.3 | 8/8 | 3.3 (0.8) |
| 3 votes (K=2) | 99% | -14.3 (0.6) | +0.0 | 100.0 | 8/8 | -1.9 (0.2) |
| 5 votes (K=4) | 95% | 42.5 (7.7) | -4.1 | 95.5 | 4/8 | 1.5 (0.8) |
| 5 votes (K=4) | 99% | -17.0 (19.9) | -0.4 | 99.5 | 7/8 | -2.7 (1.5) |

### Table H. Number of votes, MMLU-Pro only
| Probe | Target | Savings % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift (pp) |
|---|---|---|---|---|---|---|
| 2 votes (K=1) | 95% | 10.7 (33.6) | -1.9 | 97.9 | 6/8 | -0.2 (1.5) |
| 2 votes (K=1) | 99% | -7.2 (0.4) | +0.0 | 100.0 | 8/8 | -1.0 (0.1) |
| 3 votes (K=2) | 95% | 56.6 (3.9) | -4.5 | 94.9 | 4/8 | 3.3 (0.8) |
| 3 votes (K=2) | 99% | -10.7 (0.5) | +0.0 | 100.0 | 8/8 | -1.5 (0.1) |
| 5 votes (K=4) | 95% | 50.0 (8.4) | -5.1 | 94.3 | 4/8 | 1.9 (1.2) |
| 5 votes (K=4) | 99% | -17.9 (0.8) | +0.0 | 100.0 | 8/8 | -2.5 (0.2) |