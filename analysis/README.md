# Participant level uncertainty and exploratory external transfer

Revision analysis 2026-11-17. All effects are paired D minus G MAE in bpm; negative favors D. Existing frozen predictions and model selections are unchanged. These are newly requested descriptive statistical analyses, not new model evaluations.

| Study | Objective | People | Mean difference | Conditional descriptive 95% resampling interval |
|---|---|---:|---:|---|
| final | logcosh | 3 | +0.059428 | [-0.035484, +0.179426] |
| posthoc | logcosh | 12 | +0.258720 | [-0.219033, +0.716914] |
| posthoc | MSE | 12 | -0.530252 | [-0.869477, -0.211599] |
| posthoc | MAE | 12 | -0.722030 | [-1.474588, -0.091472] |
| posthoc | Huber5 | 12 | -0.023078 | [-0.506487, +0.465121] |
| external | logcosh | 8 | -9.156431 | [-18.275958, +1.785544] |

## Algorithm and interpretation

For each participant, first average the three paired seed-specific differences d_s = (1/3) sum_k (MAE_D,s,k − MAE_G,s,k). Estimate the effect by the unweighted mean of d_s over n participants. For each of 200,000 resamples, draw n participant indices uniformly with replacement and average the associated d_s values. Both methods and all three seeds remain paired within each resampled participant. Report empirical 2.5th and 97.5th percentiles using NumPy quantile(method="linear"). No seeds, windows, records or folds are counted as extra people. No participant is omitted based on error; no loss/seed winner is selected.

Reproducibility: NumPy 2.4.2, explicit Generator(PCG64). Integer seeds are2026111701 (final),2026111702 (posthoc),2026111703 (external); index-array SHA256 values are in STATISTICAL_RESULTS.json. All four posthoc objectives use the same resampled participant indices to preserve their alignment. Intervals are individual marginal summaries, not simultaneous/multiplicity-adjusted intervals. No p-values, equivalence margins or significance stars are supplied.

All intervals are conditional descriptive participant-resampling intervals given the fitted models, the three fixed optimization seeds, the observed participants and the chosen reference pipeline. They do not capture retraining variability or all study-design uncertainty. The final set has only three people: its empirical distribution has just10 unordered count configurations (27 ordered samples). The nominal95% endpoints equal the minimum and maximum observed participant-average effects; increasing Monte Carlo draws cannot make this a reliable population interval. An exact inverse-CDF enumeration of the27 ordered resamples independently checks those endpoints. The interpolation used for a finite Monte Carlo sample must not be confused with interpolating27 atoms of the exact probability distribution.

Posthoc folds share training participants and fitted models; S1–S12 were previously used in development. Treating their observed residual effects as exchangeable resampling units does not remove model-induced/cross-fold dependence. Thus the posthoc intervals are not confirmatory population confidence intervals. The MSE and MAE intervals lie below zero under this conditional procedure, which must not be presented as independently confirmed efficacy, a corrected significance finding or a reason to choose a loss and retest consumed final/external records. Crossing zero in other intervals does not establish equivalence or no effect.

## External transfer is strictly exploratory

The external eight-person result remains a frozen, nominal-alignment, cross-device transfer description. No alignment/offset/scale sensitivity analysis was run. The bootstrap interval [−18.275958,+1.785544] bpm includes zero, but even a narrower interval could not account for unknown alignment error, reference-definition differences, clock drift or domain mismatch. Do not characterize it as a demonstrated robustness/generalization advantage or clinical accuracy. The estimate is conditioned on the existing adapter and recorded reference alignment.

wrist_eight_people_G_D_seed_ranges.csv contains every actual participant ID (1,2,3,4,5,6,8,9), windows/model, G/D mean and min/max over the same three seeds, all six method-seed MAEs, and paired difference mean/range and seed values. Seed ranges are descriptive min/max, not confidence bounds. A paired difference range is computed from matched differences, not from subtracting unrelated endpoints. Subject4 worsens in each seed; the all-person mean also reverses in seed42. No person or adverse seed was removed.

| Wrist participant | G mean [seed min,max] | D mean [seed min,max] | Paired D−G mean [seed min,max] |
|---|---|---|---|
| 1 | 44.4399 [23.3079,70.3465] | 27.1726 [19.7920,33.1685] | -17.2673 [-37.1781,-3.5159] |
| 2 | 34.4925 [25.3203,48.8640] | 18.1980 [12.0902,30.0404] | -16.2945 [-36.4007,+0.7472] |
| 3 | 27.1954 [19.8218,39.4358] | 17.4533 [8.4151,31.0414] | -9.7421 [-31.0207,+8.7126] |
| 4 | 41.8298 [22.0036,75.6136] | 65.1275 [46.2223,100.7978] | +23.2977 [+18.3501,+26.3588] |
| 5 | 27.2887 [17.9526,37.9545] | 26.8474 [13.4486,52.3603] | -0.4412 [-12.5103,+14.4059] |
| 6 | 29.4399 [15.1600,41.7668] | 19.4541 [12.9606,26.5207] | -9.9857 [-28.8061,+11.3607] |
| 8 | 47.3262 [26.6207,67.8570] | 17.1006 [8.2395,23.8370] | -30.2256 [-48.6317,-18.3812] |
| 9 | 27.3299 [6.3863,47.0936] | 14.7372 [7.9899,27.9624] | -12.5927 [-38.8344,+21.5761] |

## What the training records establish

Original G/D records for all six method-seed runs were read from the preserved source histories and matched to their source-index hashes. Their actual stops were G42 117epochs/best97; D42 100/best80; G43 73/best53; D43 74/best54; G44 69/best49; D44 80/best60. All stopped after20 validation checks without strict improvement, below the200epoch ceiling. Their best-update/selected validation logcosh and terminal update counts agree with summaries. Full existing curves are reproduced as numbers in original_G_D_full_histories.csv, with stops/SHA in original_G_D_stopping.csv. No curve was smoothed or truncated.

Posthoc stopping reuses the independently checked90-record receipts and79+11 result tables; counts: {"inner_early_stop": 90}. This is not a second90history or prediction audit. Original selection used equal-person validation logcosh; posthoc used equal-person inner MAE. They are different selection contracts.

The records verify that the specified stopping rules executed, not that training was sufficient. A plateau cannot rule out undertraining, suboptimal optimizer/learning rate, or improvement under a different schedule. Training curves average minibatches while model parameters evolve; validation curves evaluate the epoch-end model and weight people equally. Those curves differ in model state, timing, population and weighting, so their gap is not a fixed-model generalization gap. No longer-training or schedule-sweep experiment is authorized/performed here. Keep “undertraining cannot be excluded” as a limitation; do not label a flat curve proof of convergence.

## Inputs and deliverables

Fixed final/external aggregate SHA values match their original independent audit receipts; posthoc person-table SHA matches OUTER_RESULTS_RECEIPT_20261112.json. Source and generated-artifact hashes are in JSON records and ARTIFACT_MANIFEST.json. participant_means.csv has59 participant-condition rows, participant_seed_pairs.csv177 paired seed rows, participant_bootstrap95.csv six contrasts, plus the complete external-eight table and training records. analyze_revision.py recreates these small-table calculations using the existing environment; it imports no Torch and reads no prediction CSV or raw signal. Only review/ is written.

For the English revision, use the table/algorithm above and the strict exploratory/conditional qualifications verbatim where appropriate. No training, inference, repeat final test, alignment retuning or new data were used.
