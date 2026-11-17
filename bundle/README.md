# Local reproducibility bundle, revision 2026-11-17

This local ZIP is prepared for sharing; it is not uploaded and has no permanent public URL. It contains actual frozen numerical training, inference and collector sources, configurations, fixed initial backbones (3 x39,498 bytes), subject splits, development eligible-window IDs, committed unsmoothed histories, and already-fixed development outer predictions/summaries for90 arms. No raw signals, trained/best/resume checkpoints, final-test arrays or final-test predictions are included. No new model evaluation or training was performed to build it.

## Verified portable commands (no GPU or external data required)

From the extracted bundle directory, with Python3.13:

    python tools/verify_bundle.py --root .
    python tools/reanalyse_fixed_predictions.py --root . --output reanalysis_output

The second command streams the fixed development prediction CSVs and recomputes subject/seed MAE and equal12-subject seed aggregates, checking against the original aggregate to tolerance1e-5 bpm. It does not run inference. Result directories are newly created and do not change archived evidence. Stopping statistics/curves are in stopping/; raw per-arm histories remain in evidence/development_cv/. Never compare raw training objective magnitudes across different losses. Fold-level mean/median stopping summaries are descriptive only. Shared Adam/patience do not prove full convergence or isolate the effect of loss.

## Rebuilding data and numerical experiments (documented, NOT end-to-end verified here)

PPG-DaLiA public dataset source as recorded during acquisition:
https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip
Author dataset page: https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html
Recorded author-linked archive: https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download
These historical source links were not fetched or availability-tested during this revision. Data licensing/access terms apply separately; upstream model Apache2.0 license is retained.

Fetch raw data independently. For each S1..S12.pkl verify SHA256 against experiments/data/manifests/S*_recovery.json (field sha256) and CRC/length metadata. Historical transport used exact member recovery from an incomplete archive, not a falsely claimed verified full archive. Do not assert a complete archive SHA. Set recovery output paths to the actual local member files, preserving expected raw SHA. Original data preparation code:

    python experiments/preprocessing/prepare_recovered_subject.py --subject 1

Repeat only S1..S12. Validate resulting eligible rows against data/eligible_development_windows.csv and expected hashes in data/eligible_source_bindings.json. Preprocessing uses wrist BVP64Hz[::2], ACC32Hz,256-sample windows/hop64, no normalization, original label-index mapping; details in the retained source. Per-fold train/inner/outer subject IDs are in data/splits.json. Original train/val prepared NPZ SHA bindings and feature-builder numerical code are included, but their large cached arrays are omitted. Rebuild the frontend/features with round2_v1_4/executor and implementation source following its protocol; validate hashes/identities before running workers.

Training implementation: experiments/candidate_versions/supplemental_cv_loss_v1_1/{run.py,resumable.py,checkpoints.py}; numerical core/factory/collector: supplemental_cv_loss_v1/{engine.py,run.py}; feature/gate implementation: round2_v1_4/implementation; backbone: experiments/models/qppg_fixed.py and licensed third-party TimePPG_float.py. Inference source/config: wrist_external_v1 and final_evaluation_v1 (source only, if present). Actual recovery runtime: autonomous_owner_v1. Default verification is not equivalent to executing a new experiment.

Archived worker invocation (on a separately constructed, reviewed local study, not the archived completed study):

    python experiments/candidate_versions/autonomous_owner_v1/worker.py --worker <NEW_BUNDLE> --index <0..89> --owner <NEW_OWNER_CONTROL>

Archived scheduler, callbacks, Windows locks, resource gates and path bindings are historical execution infrastructure. Absolute personal paths were normalized and some cross-package bindings point to deliberately omitted caches/checkpoints. Original FREEZE hashes are retained as provenance and will not validate every normalized distribution file. Rebind only a NEW copy, regenerate its derivative freeze/config paths and independently review it before training; do not bypass identity assertions on completed originals. No claim of one-command end-to-end retraining or cross-platform scheduler equivalence is made. Archived96h/per-arm2h policy was later superseded by recorded execution overrides (finite cumulative120 worker-hours; no per-arm2h cap), without changing optimizer, data, RNG or epoch/patience. Full Adam/RNG continuation cannot be reproduced from this lightweight bundle because trained/resume states are intentionally omitted.

Existing S13..S15 and external-test outcomes were already consumed in the original project; this bundle does not authorize retuning or reopening them. The90-arm study is retrospective development robustness, not a new untouched final test. All90 training checkpoints were fixed before its registered outer evaluation. Selected models/large checkpoints are identified by historical commit hashes but not distributed here. Final-test metrics are not recomputed by the verified commands.

## Integrity and scope

SHA256_MANIFEST.json binds packaged bytes; provenance/source_map.json maps original and packaged SHA and path normalization. Original completed artifacts were read only. Requirements and historical training environment are under environment/. The tested path is integrity + fixed-prediction reanalysis. Full data download, cache rebuilding, training, original collector execution and fresh inference were not executed in this revision. No permanent artifact URL is invented.

Source audit note: all24 bindings of the actual v1.1 math freeze, all49 round2_v1_4 bindings, inference code bindings and the actual runtime freeze matched current original bytes. The older supplemental_cv_loss_v1 freeze has one document-only mismatch: the research protocol Markdown has evolved since that earlier freeze. This is retained explicitly in provenance/FROZEN_SOURCE_CHECKS.json; no old protocol hash is relabeled as a verified match. v1.1 numerical source bindings have no mismatch.
