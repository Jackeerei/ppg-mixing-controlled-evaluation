# Controlled comparison of raw and filtered PPG inputs

This repository accompanies a wrist heart-rate estimation study comparing an input-dependent mixing weight (D) with a learned global weight (G) and three input controls.

## Recalculate the main gate comparison

From a repository checkout, run `python analysis/reproduce_locked_gate_table.py`. The script uses the public participant-by-seed G and D errors. It reconstructs the nine paired differences, their participant means and the equal-person result (+0.0594 bpm). The three person means range from −0.0355 to +0.1794 bpm. The additional t interval assumes independent normal person effects. This calculation starts from archived participant summaries. Window-level scoring for these three people requires the original predictions.

The [development comparison](analysis/20261119/existing_outer_D_minus_G_summary.csv) uses predictions from the selected checkpoint. The [common-epoch diagnostic](analysis/20261119/same_epoch_inner_within_fold_summary.csv) compares inner-validation errors at the same epoch. Epoch 27 is the last epoch observed for every pair. These two analyses use different evaluation data.

Run `python analysis/check_common_epochs.py PATH_TO_EXTRACTED_BUNDLE` to check the common-epoch table against the 72 published D/G histories.

The [current English supplement (PDF)](supplement/20261119/supplement_revised_clean.pdf) and [editable Word version](supplement/20261119/supplement_revised_clean.docx) describe the experiments and the limits of each check.

## Download and reproduce

Download [the complete code and evidence bundle](bundle/reproducibility_bundle_20261117.zip) and extract it. Its SHA256 is:

```
8542f3bc87f845759835708c8db341ffd39b22a4ba6ac821a3a9c7ef42d14248
```

The archive contains 688 files: actual numerical source, configurations, split IDs, 51,690 development eligible-window IDs, small initial backbones, all 90 committed training histories, and fixed development outer predictions/summaries. Raw signals and trained/resume checkpoints are omitted. [Detailed archive instructions and limitations](bundle/README.md).

From the extracted bundle directory, with Python 3.13:

```sh
python tools/verify_bundle.py --root .
python tools/reanalyse_fixed_predictions.py --root . --output reanalysis_output
```

These commands were executed on an extracted copy. Integrity passed; recomputed 90-arm participant/seed summaries differed from archived values by at most 3.56e-15 bpm. They reanalyse fixed predictions and do not run model inference. Full raw-data acquisition, retraining and new inference were not verified end to end. Historic freezes bind original files; normalized distribution paths have a separate provenance map. See the archive README before attempting retraining in a new copy.

The numerical implementation and configurations can also be browsed under [code/](code/). The ZIP remains the complete canonical distribution with data bindings and provenance.

## Participant-level analyses

[Analysis data and explanation](analysis/) include all eight external participants' G/D means and seed ranges, all six principal D-minus-G descriptive intervals, and paired participant/seed inputs. With NumPy 2.4.2 installed:

```sh
python analysis/reproduce_intervals.py
```

Each participant's three paired seed effects is first averaged, then people are sampled with replacement. The 200,000-draw intervals are conditional descriptive summaries: n=3 is too small for reliable population coverage; post-hoc folds share training data/models; external nominal alignment was not stress-tested. No equivalence, confirmatory significance or demographic fairness is established.

The 90 histories report actual early stopping at 27–173 epochs. Curves and stopping summaries do not exclude undertraining or objective/optimizer interaction.

## Locked and external archive checks

The [18 November audit materials](audit/20261118/) give deterministic checksums and a standard-library verifier for candidate, eligible, and successful saved-window coverage. The directory also records the historical review procedure and all nine locked D−G participant/seed differences. Checking original bytes requires the retained archive; the release does not provide full held-out/external predictions or demonstrate end-to-end retraining. [Fixed audit snapshot](https://github.com/Jackeerei/ppg-mixing-controlled-evaluation/tree/640089cd7b4915b711b0eaab245969955dfc2d58/audit/20261118).

## Earlier supplement versions

The English supplement revised on 18 November is available as [PDF](supplement/20261118/supplement_revised_clean.pdf) and [editable Word](supplement/20261118/supplement_revised_clean.docx). It includes the archived-run audit procedure and limitations. The [17 November supplement](supplement/supplement_revised_clean.pdf) remains accessible. The fixed implementation and development-analysis snapshot is b598d26f2b60b0588810d8fdbee488401e40d49e.

## Data and licenses

Public dataset acquisition links and raw-member hashes are documented in the archive; raw participant signals are not redistributed. Dataset terms apply separately. Upstream source license/notice files remain in their original package locations. This repository does not impose a new blanket license on third-party code, data, predictions or manuscript material; retain the included notices and consult each source's terms.
