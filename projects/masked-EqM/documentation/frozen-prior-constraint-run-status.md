# Frozen-prior constraint run status

## Pre-launch

* Immutable evaluator revision: `9d09905`.
* Locked manifests: pilot `eb0e5f3343be7f74c052a8b5fd62cc062b905c64deb85b7fdaa5448e71daf90f`; final `34a4540e0a0739a68590d2d9e0f1ca73e11e64fa01275d12d7fdbfe3d72a90ec`.
* Focused tests: 7 passed locally, including exact tensor-level disabled-projection GD/NAG regression.
* Submission attempt: none created. Scheduler rejected the 9-task smoke array with `QOSMaxSubmitJobPerUserLimit`; four unrelated direct-energy/FID jobs were active. They were not modified.

The smoke will be resubmitted unchanged once capacity is available.
