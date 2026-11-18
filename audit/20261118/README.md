# Locked and external evaluation audit records

This release adds verification metadata for the archived PPG-DaLiA held-out test and exploratory Wrist transfer. It leaves the previously published implementation, predictions from development folds, and analysis unchanged.

| Archive | Model runs | Candidate windows per run | Eligible windows per run | Successful saved predictions per run |
| --- | ---: | ---: | ---: | ---: |
| Locked PPG-DaLiA | 21 (15 original and 6 separate ablation runs) | 13,007 | 13,007 | 13,007 |
| Exploratory Wrist transfer | 15 | 3,332 | 3,320 | 3,320 |

[Checksums](LOCKED_ARCHIVE_CHECKSUMS.json) bind the original preparation and result manifests, candidate-window CSVs, model and protocol records, and saved outputs. They include deterministic digests of candidate, eligible, and successful window identities and availability flags. Per-window values and full CSVs, original signals, trained weights, and other source bytes are not distributed here. The manifest's release-scope note describes its local status at generation; this directory publishes that verified metadata unchanged.

The [standard-library verifier](verify_locked_archives.py) can check byte identity and window coverage if the reader already has a matching copy of the archived files. Use Python 3.11 or later. Supply the directory containing the archived `reports/` and `results/` trees:

```sh
python verify_locked_archives.py --archive-root /path/to/experiments
```

Against the retained archive, the checker verified 86 files and all 36 runs. The published files alone do not supply the archived inputs needed to execute that check. No waveform processing, training, inference, or error-metric recalculation is performed by the verifier.

## Independent historical review

The reviewer used the frozen preparation, model, and result records as references; verified window identities, eligibility and saved-output coverage; and separately recomputed the participant and paired summaries from archived predictions. The held-out review covered all 273,147 saved rows and all 45 paired cells across the original and distinct ablation comparisons. The external review covered 49,980 candidate rows, including 49,800 eligible predictions. The historical reviews also checked a limited CPU replay of 24 held-out and 40 external model/window pairs. Those replays were run in the original review, not in this revision.

The [audit index](HISTORICAL_AUDIT_INDEX.json) identifies retained review scripts, logs, machine evidence, reports and original result manifests and provides fingerprints where recorded. Its historical relative paths identify items in the retained research archive; those original files and replay inputs are not implied to be downloadable from this directory.

The nine locked adaptive-minus-global comparisons are available as [participant/seed aggregates](locked_final_D_minus_G_nine_cells.csv). They are repeated fits on three participants. Negative D−G favors the adaptive gate. The repository's [development reanalysis and participant intervals](../../analysis/) can be run on the published inputs separately.

Byte and coverage checks do not establish raw signal/label alignment, regenerate the missing archive, or demonstrate end-to-end retraining or reconstruction. External transfer remains strictly exploratory under nominal alignment.
