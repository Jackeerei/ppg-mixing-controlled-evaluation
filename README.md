# PPG mixing under matched controls

Research code, fixed development predictions, participant-level analyses and supplementary material for **Raw–Filtered PPG Mixing Under Matched Controls: Held-Out and Post-hoc Subject-Wise Evaluation**.

The contribution is a controlled evaluation of input construction with a shared regressor architecture. The results do not establish stable adaptive-gate superiority or an isolated intrinsic loss effect. Three optimization seeds are repeated model fits, not extra participants.

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

## Supplement

The 19-page English supplement is available as [PDF](supplement/supplement_revised_clean.pdf) and [editable Word](supplement/supplement_revised_clean.docx). The manuscript availability statements identify this repository and the fixed code/analysis snapshot b598d26f2b60b0588810d8fdbee488401e40d49e. It contains detailed definitions, participant tables and all stopping diagnostics. The conference's acceptance/publication is not implied.

## Data and licenses

Public dataset acquisition links and raw-member hashes are documented in the archive; raw participant signals are not redistributed. Dataset terms apply separately. Upstream source license/notice files remain in their original package locations. This repository does not impose a new blanket license on third-party code, data, predictions or manuscript material; retain the included notices and consult each source's terms.
