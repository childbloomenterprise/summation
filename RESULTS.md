# Results

Frozen evaluation run once against `PREREG.md` (RSICD) and `PREREG_ROBOT.md` (DROID). All numbers are on held-out test sets, percent, mean ± std over seeds 0/1/2. CPU only (8 cores). Tables regenerate with `python demo/report.py demo/results.json` / `demo/results_droid.json`.

## Verdict

| | RSICD (satellite) | DROID (robot camera) |
|---|---|---|
| Stitch N=1000 beats CLIP-B/32 | PASS (18.2 vs 16.4) | PASS (14.4 vs 5.6) |
| Stitch N=1000 beats SigLIP-B/16 | **FAIL** (18.2 vs 22.0) | PASS (14.4 vs 8.0) |
| Stitch N=100 beats CLIP-B/32 | **FAIL** (10.8) | **FAIL** (4.2) |
| Within 5 mR of RemoteCLIP (stretch) | **FAIL** (18.2 vs 33.8) | n/a |
| Zero-pair t2i R@10 ≥ 5× chance | **FAIL** (0.6 vs ≥4.6) | **FAIL** (1.4 vs ≥5.7) |
| Pre-registered verdict | beats old CLIP only; not worth a product | worth a pilot with a robot-data team |

## RSICD (1,093 test images, 5,465 captions)

| method | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 | mR | mutual-10NN / fit time |
|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5.5 | 17.7 | 27.9 | 5.3 | 16.0 | 26.3 | 16.4 | 0.183 |
| clip-l14 | 5.0 | 19.4 | 30.8 | 6.3 | 19.5 | 29.7 | 18.5 | 0.199 |
| siglip-b16 | 7.1 | 23.1 | 34.9 | 9.3 | 23.5 | 33.8 | 22.0 | 0.191 |
| remoteclip-b32 (in-domain ceiling) | 10.5 | 32.3 | 48.6 | 16.7 | 40.0 | 54.5 | 33.8 | 0.243 |
| stitch primary N=100 | 2.5 ±0.2 | 9.9 ±0.4 | 18.3 ±1.3 | 3.7 ±0.1 | 11.6 ±0.5 | 18.6 ±1.1 | 10.8 ±0.6 | native 0.192 / 2.9s |
| stitch primary N=300 | 3.3 ±0.3 | 13.9 ±0.6 | 24.9 ±0.8 | 6.2 ±0.7 | 17.9 ±0.6 | 27.8 ±1.1 | 15.7 ±0.5 | 3.7s |
| stitch primary N=1000 | 4.1 ±0.2 | 16.3 ±0.3 | 28.3 ±0.4 | 8.8 ±0.5 | 20.0 ±0.5 | 31.5 ±1.1 | 18.2 ±0.3 | 4.1s |
| stitch secondary N=100 | 2.0 ±0.1 | 8.4 ±0.8 | 15.5 ±1.2 | 2.9 ±0.4 | 10.5 ±1.4 | 17.2 ±2.3 | 9.4 ±1.0 | native 0.172 / 1.3s |
| stitch secondary N=300 | 2.5 ±0.3 | 11.1 ±0.3 | 20.0 ±0.9 | 5.1 ±1.7 | 14.8 ±1.5 | 23.2 ±2.7 | 12.8 ±1.2 | 1.8s |
| stitch secondary N=1000 | 2.9 ±0.1 | 13.0 ±0.2 | 23.3 ±0.2 | 8.1 ±0.1 | 19.3 ±0.6 | 29.3 ±0.9 | 16.0 ±0.3 | 1.7s |
| zero-pair primary | 0.1 ±0.0 | 0.3 ±0.2 | 0.6 ±0.3 | 0.1 ±0.0 | 0.3 ±0.1 | 0.4 ±0.3 | 0.3 ±0.1 | 8s |
| exploratory: CLIP-B/32 + rotation N=1000 | 4.0 ±0.2 | 16.0 ±0.3 | 27.5 ±0.3 | 6.7 ±0.5 | 19.6 ±0.6 | 31.0 ±0.9 | 17.5 ±0.4 | |

## DROID (872 test episodes, 2,616 instructions; fit pool 1,692 episodes)

| method | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 | mR | mutual-10NN / fit time |
|---|---|---|---|---|---|---|---|---|
| clip-b32 | 2.1 | 5.9 | 10.5 | 1.8 | 5.3 | 8.1 | 5.6 | 0.061 |
| clip-l14 | 2.9 | 9.6 | 15.3 | 2.1 | 8.3 | 11.2 | 8.2 | 0.069 |
| siglip-b16 | 2.3 | 9.0 | 13.4 | 2.4 | 8.8 | 11.8 | 8.0 | 0.081 |
| stitch primary N=100 | 1.1 ±0.6 | 4.3 ±2.0 | 7.9 ±2.7 | 1.4 ±0.7 | 4.2 ±1.2 | 6.4 ±2.1 | 4.2 ±1.5 | native 0.085 / 6.8s |
| stitch primary N=300 | 2.9 ±0.2 | 10.0 ±0.9 | 16.3 ±1.5 | 3.5 ±0.5 | 9.4 ±1.0 | 13.7 ±1.1 | 9.3 ±0.8 | 14.9s |
| stitch primary N=1000 | 5.0 ±0.7 | 16.2 ±1.0 | 23.3 ±0.4 | 5.3 ±0.2 | 15.3 ±0.4 | 21.5 ±1.3 | 14.4 ±0.6 | 8.4s |
| stitch secondary N=100 | 0.9 ±0.2 | 3.4 ±0.8 | 6.0 ±1.6 | 0.7 ±0.2 | 2.8 ±0.6 | 5.5 ±1.3 | 3.2 ±0.7 | native 0.079 / 2.5s |
| stitch secondary N=300 | 2.7 ±0.3 | 8.9 ±0.8 | 13.9 ±1.3 | 3.0 ±0.7 | 8.9 ±0.8 | 13.3 ±0.6 | 8.4 ±0.7 | 2.7s |
| stitch secondary N=1000 | 4.2 ±0.4 | 14.2 ±0.5 | 20.9 ±0.6 | 5.4 ±0.5 | 13.8 ±1.3 | 20.1 ±1.4 | 13.1 ±0.7 | 4.2s |
| zero-pair primary | 0.1 ±0.0 | 0.6 ±0.1 | 1.4 ±0.1 | 0.2 ±0.1 | 0.6 ±0.1 | 1.0 ±0.0 | 0.7 ±0.0 | 60s |
| exploratory: CLIP-B/32 + rotation N=1000 | 4.1 ±0.2 | 12.1 ±0.4 | 17.5 ±0.4 | 4.5 ±0.1 | 12.4 ±0.9 | 17.6 ±0.7 | 11.4 ±0.1 | |

Fit time includes 5-fold CV over the PCA dimension k.

## What this does and does not show

Verified:
- Few-pair stitching is not a free lunch. At N=100 it loses to zero-shot CLIP in both domains; it needs roughly 300–1,000 in-domain pairs to win.
- Pair-free alignment (Gromov-Wasserstein + Wasserstein-Procrustes) failed in both domains; results are near chance. The `--unpaired` mode should be treated as experimental.
- Satellite: SigLIP and RemoteCLIP (free, Apache) both beat stitching. No product case there.
- Robot frames: off-the-shelf CLIP/SigLIP are weak (mR 5.6–8.2); stitching with 1,000 pairs gets 14.4, ~1.8× SigLIP.
- Exploratory (not pre-registered): rotating CLIP-B/32 itself with the same 1,000 pairs gets 11.4 on DROID and 17.5 on RSICD. So most of the DROID gain comes from fitting on in-domain pairs at all; swapping in DINOv2+MPNet adds a further ~3 mR.

Caveats / inference:
- DROID fit and test come from adjacent data files (episodes 0–~2,900 of the first metadata file). The same labs, scenes and task phrasings may appear in both; 18.7% of test instructions occur verbatim in the fit pool. A lab-held-out split could lower the stitched numbers. This was not tested.
- DROID instructions are short and repetitive ("put the marker in the cup"); retrieval is scored against one exact episode, so absolute numbers are low for every method.
- One middle frame per episode, one camera. Clip-level (video) search was not tested.

Deviations from pre-registration:
- RSICD and DROID: none in protocol. For DROID, train-split embeddings were computed only for the stitched encoders and CLIP-B/32 (baselines only need test); an extraction run was restarted once with this change.
