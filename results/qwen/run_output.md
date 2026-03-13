# 500 Scenario Experiment - COMPLETE RESULTS

## Status: COMPLETED SUCCESSFULLY (23 minutes)

## TF-IDF Text Baseline
- Text baseline accuracy: 0.862
- Deception probe accuracy: 0.937
- Probe beats text baseline: YES ✓

## WHITE LIES vs SERIOUS LIES ANALYSIS
- White lies — lying score: mean=0.9635, std=0.1677
- White lies — honest score: mean=-0.0480, std=0.1819
- Serious lies — lying score: mean=0.9842, std=0.0913
- Serious lies — honest score: mean=0.0093, std=0.0679
- White vs Serious lying scores: t=-1.331, p=0.1838
- Significantly different: NO (great! probe treats them the same)
- White lie detection accuracy: 95/100 (95.0%)

## FINAL SUMMARY - 500 SCENARIO EXPERIMENT
- Model: Qwen/Qwen2.5-3B-Instruct
- Method: Generation-based with length control
- Layer: 10
- Scenarios: 435 (348 train / 87 test)
- Samples: 870 (696 train / 174 test)
- Categories: 16

## MAIN RESULTS
- CV accuracy: 0.958
- Held-out accuracy: 0.937
- False positives: 5
- False negatives: 6
- P-value: 0.0000
- TF-IDF baseline: 0.862

## LENGTH CONTROLS (ALL 5 PASSED!)
1. Length-score correlation: r=0.012 (p=0.7214) — NEGLIGIBLE correlation!
2. Truncation test (20 tokens): 0.931 — Still works with only 20 tokens!
3. Residualized probe: 0.931 — Still works after removing length variance!
4. Length-only baseline: 0.517 — Length alone is USELESS (barely above chance)!
5. Short/Long subsets: 0.933 / 0.941 — Works equally well on short AND long responses!

## WHITE LIES
- White lie detection: 95.0%
